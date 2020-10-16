import numpy as np
seed = 123
np.random.seed(seed)
from tensorflow import set_random_seed
set_random_seed(seed)

import keras
from sklearn.metrics import classification_report
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers import Input, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pandas as pd
import utility as ut
from sklearn import preprocessing

def get_image_size(num_col):
    matx = 2
    i = False
    while i == False:
        size = matx * matx
        if size >= num_col:
            padding = size-num_col
            i = True
        else:
            matx = matx + 1
    return matx, padding

def inception_module(layer_in, f1, f2, f3):
    # 1x1 conv
    conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)
    # 3x3 conv
    conv3 = Conv2D(f2, (3, 3), padding='same', activation='relu')(layer_in)
    # 5x5 conv
    conv5 = Conv2D(f3, (5, 5), padding='same', activation='relu')(layer_in)
    # 3x3 max pooling
    pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_in)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out

namedataset = "receipt"
df = pd.read_csv('kometa_fold/'+namedataset+'feature.csv', header=None)
num_col = df.iloc[:, :-1] # remove target column
num_col = len(df. columns)
fold1, fold2, fold3 = ut.get_size_fold(namedataset)
target = df[df.columns[-1]]
df_labels = np.unique(list(target))

img_size, padding = get_image_size(num_col)

label_encoder = preprocessing.LabelEncoder()
integer_encoded = label_encoder.fit_transform(df_labels)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
onehot_encoder.fit(integer_encoded)
onehot_encoded = onehot_encoder.transform(integer_encoded)

train_integer_encoded = label_encoder.transform(target).reshape(-1, 1)
train_onehot_encoded = onehot_encoder.transform(train_integer_encoded)
y_one_hot = np.asarray(train_onehot_encoded)

n_classes = len(df_labels)

Y_1 = y_one_hot[:fold1]
Y_2 = y_one_hot[fold1:(fold1+fold2)]
Y_3 = y_one_hot[(fold1+fold2):]

dense1 = 64
dense2 = 128
dropout1 = 0.3
dropout2 = 0.4
learning_rate = 0.0002

f1, f2, f3 = 64, 128, 32
for f in range(3):
    print("Fold n.", f)
    if f == 0:
        y_a_train = np.concatenate((Y_1, Y_2))
        y_a_test = Y_3
    elif f == 1:
        y_a_train = np.concatenate((Y_2, Y_3))
        y_a_test = Y_1
    elif f == 2:
        y_a_train = np.concatenate((Y_1, Y_3))
        y_a_test = Y_2

    X_a_train = np.load("image/receipt/receipt_train_fold_"+str(f)+".npy")
    X_a_test = np.load("image/receipt/receipt_test_fold_"+str(f)+".npy")

    X_a_train = np.asarray(X_a_train)
    X_a_test = np.asarray(X_a_test)
    X_a_train = X_a_train.astype('float32')
    X_a_train = X_a_train / 255.0

    X_a_test = X_a_test.astype('float32')
    X_a_test = X_a_test / 255.0
    layer_in = Input(shape=(int(img_size), int(img_size), 3))
    layer_out = inception_module(layer_in, f1, f2, f3)
    layer_out = inception_module(layer_out, f1, f2, f3)
    layer_out = Dense(dense1, activation='relu', kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(layer_out)
    layer_out = Dropout(dropout1)(layer_out)
    layer_out = Flatten()(layer_out)
    layer_out = Dense(dense2, activation='relu', kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(layer_out)
    layer_out = Dropout(dropout2)(layer_out)

    optimizer = Adam(lr=learning_rate)

    out = Dense(n_classes, activation='softmax')(layer_out)
    model = Model(inputs=layer_in, outputs=out)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model_checkpoint = ModelCheckpoint("dataset/" + namedataset + "/" + 'model_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    model.fit(X_a_train, y_a_train, epochs=200, batch_size=128, verbose=1, callbacks=[early_stopping, lr_reducer], validation_split=0.2)
    model.save("dataset/" + namedataset + "/" + namedataset + ".h5")

    # evaluate
    print('Evaluating final model...')
    preds_a = model.predict(X_a_test)

    y_a_test = np.argmax(y_a_test, axis=1)
    preds_a = np.argmax(preds_a, axis=1)

    print(classification_report(y_a_test, preds_a, digits=3))

