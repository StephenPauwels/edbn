import numpy as np
seed = 123
np.random.seed(seed)
from tensorflow import set_random_seed
set_random_seed(seed)

import keras
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers import Input, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
import pandas as pd
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from time import perf_counter
import time
import os
import utility as ut
from sklearn import preprocessing

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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


def get_model(dense1, dense2, dropout1, dropout2, n_classes, learning_rate):
    inputs = Input(shape=(img_size, img_size, 3))
    filters = (inception_module(inputs, 64, 128, 32))
    filters = (inception_module(filters, 64, 128, 32))

    layer_out = Dense(dense1, activation='relu', kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(filters)
    layer_out = Dropout(dropout1)(layer_out)
    layer_out = Flatten()(layer_out)
    layer_out = Dense(dense2, activation='relu', kernel_initializer=keras.initializers.glorot_uniform(seed=seed))(layer_out)
    layer_out = Dropout(dropout2)(layer_out)

    optimizer = Adam(lr=learning_rate)

    out = Dense(n_classes, activation='softmax')(layer_out)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    return model


def fit_and_score(params):
    print(params)
    start_time = perf_counter()

    model = get_model(learning_rate=params['learning_rate'], dense1=params['dense1'],dense2=params['dense2'],
                      dropout1=params['dropout1'], dropout2=params['dropout2'], n_classes=params['n_classes'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)

    h = model.fit(X_a_train, y_a_train, epochs=200, verbose=0, validation_split=0.2, callbacks=[early_stopping], batch_size=2 ** params['batch_size'])

    scores = [h.history['val_loss'][epoch] for epoch in range(len(h.history['loss']))]
    score = min(scores)
    print(score)

    global best_score, best_model, best_time, best_numparameters
    end_time = perf_counter()

    if best_score > score:
        best_score = score
        best_model = model
        best_numparameters = model.count_params()
        best_time = end_time - start_time

    return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(h.history['loss']), 'n_params': model.count_params(),
            'time': end_time - start_time}


logfile = "receipt.csv"  # change with the name of the dataset
output_file = "receipt.log"  # change with the name of the dataset

current_time = time.strftime("%d.%m.%y-%H.%M", time.localtime())
outfile = open(output_file, 'w')

outfile.write("Starting time: %s\n" % current_time)

n_iter = 10

f1, f2, f3 = 64, 128, 32
decay = 0.0
f = 0

namedataset = "receipt"
df = pd.read_csv('kometa_fold/'+namedataset+'feature.csv', header=None)
num_col = df.iloc[:, :-1] # remove target column
num_col = len(df. columns)
fold1, fold2, fold3 = ut.get_size_fold(namedataset)
target = df[df.columns[-1]]
df_labels = np.unique(list(target))

img_size, pad = get_image_size(num_col)

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

print(n_classes)

space = {'dense1': hp.choice('dense1', [32, 64, 128]),
         'dense2': hp.choice('dense2', [32, 64, 128]),
         'dropout1': hp.uniform("dropout1", 0, 1),
         'dropout2': hp.uniform("dropout2", 0, 1),
         'batch_size': hp.choice('batch_size', [7, 8]),
         'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01)),
         'n_classes': n_classes}

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

    # model selection
    print('Starting model selection...')
    best_score = np.inf
    best_model = None
    best_time = 0
    best_numparameters = 0

    trials = Trials()
    best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=n_iter, trials=trials,
                rstate=np.random.RandomState(seed + f))
    best_params = hyperopt.space_eval(space, best)

    outfile.write("\nHyperopt trials")
    outfile.write("\ntid,loss,learning_rate,batch_size,time,n_epochs,n_params,perf_time,dense1,dense2,drop1,drop2")
    for trial in trials.trials:
        outfile.write("\n%d,%f,%f,%d,%s,%d,%d,%f,%d,%d,%f,%f" % (trial['tid'],
                                                        trial['result']['loss'],
                                                        trial['misc']['vals']['learning_rate'][0],
                                                        trial['misc']['vals']['batch_size'][0] + 7,
                                                        (trial['refresh_time'] - trial['book_time']).total_seconds(),
                                                        trial['result']['n_epochs'],
                                                        trial['result']['n_params'],
                                                        trial['result']['time'],
                                                        trial['misc']['vals']['dense1'][0],
                                                        trial['misc']['vals']['dense2'][0],
                                                        trial['misc']['vals']['dropout1'][0],
                                                        trial['misc']['vals']['dropout2'][0]
                                                        ))
    outfile.write("\n\nBest parameters:")
    print(best_params, file=outfile)
    outfile.write("\nModel parameters: %d" % best_numparameters)
    outfile.write('\nBest Time taken: %f' % best_time)

    # evaluate
    print('Evaluating final model...')
    preds_a = best_model.predict(X_a_test)

    y_a_test = np.argmax(y_a_test, axis=1)
    preds_a = np.argmax(preds_a, axis=1)

    outfile.write(np.array2string(confusion_matrix(y_a_test, preds_a), separator=", "))
    outfile.write(classification_report(y_a_test, preds_a, digits=3))

    outfile.flush()

outfile.close()
