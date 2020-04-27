import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing
from keras.layers import Conv2D, Activation
from keras import regularizers
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization

seed = 123
np.random.seed(seed)
from tensorflow import set_random_seed
set_random_seed(seed)

def get_label(act):
    i = 0
    list_label = []
    while i < len(act):
        j = 0
        while j < (len(act.iat[i, 0]) - 1):
            if j > 0:
                list_label.append(act.iat[i, 0][j + 1])
            else:
                pass
            j = j + 1
        i = i + 1
    return  list_label


def dataset_summary(dataset):
    df = pd.read_csv(dataset, sep=",")
    print("Activity Distribution\n", df['Activity'].value_counts())
    n_caseid = df['CaseID'].nunique()
    n_activity = df['Activity'].nunique()
    print("Number of CaseID", n_caseid)
    print("Number of Unique Activities", n_activity)
    print("Number of Activities", df['Activity'].count())
    cont_trace = df['CaseID'].value_counts(dropna=False)
    max_trace = max(cont_trace)
    print("Max lenght trace", max_trace)
    print("Mean lenght trace", np.mean(cont_trace))
    print("Min lenght trace", min(cont_trace))
    return df, max_trace, n_caseid, n_activity

def get_image(act_val, time_val, max_trace, n_activity):
    i = 0
    matrix_zero = [max_trace, n_activity, 2]
    image = np.zeros(matrix_zero)
    list_image = []

    while i < len(time_val):
        j = 0
        list_act = []
        list_temp = []
        cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9 = 0, 0, 0, 0, 0, 0, 0, 0, 0
        diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8, diff9 = 0, 0, 0, 0, 0, 0, 0, 0, 0
        while j < (len(act_val.iat[i, 0]) - 1):
            start_trace = time_val.iat[i, 0][0]
            if act_val.iat[i, 0][0 + j] == 1:
                cont1 = cont1 + 1
                diff1 = time_val.iat[i, 0][0 + j] - start_trace
            elif act_val.iat[i, 0][0 + j] == 2:
                cont2 = cont2 + 1
                diff2 = time_val.iat[i, 0][0 + j] - start_trace
            elif act_val.iat[i, 0][0 + j] == 3:
                cont3 = cont3 + 1
                diff3 = time_val.iat[i, 0][0 + j] - start_trace
            elif act_val.iat[i, 0][0 + j] == 4:
                cont4 = cont4 + 1
                diff4 = time_val.iat[i, 0][0 + j] - start_trace
            elif act_val.iat[i, 0][0 + j] == 5:
                cont5 = cont5 + 1
                diff5 = time_val.iat[i, 0][0 + j] - start_trace
            elif act_val.iat[i, 0][0 + j] == 6:
                cont6 = cont6 + 1
                diff6 = time_val.iat[i, 0][0 + j] - start_trace
            elif act_val.iat[i, 0][0 + j] == 7:
                cont7 = cont7 + 1
                diff7 = time_val.iat[i, 0][0 + j] - start_trace
            elif act_val.iat[i, 0][0 + j] == 8:
                cont8 = cont8 + 1
                diff8 = time_val.iat[i, 0][0 + j] - start_trace
            elif act_val.iat[i, 0][0 + j] == 9:
                cont9 = cont9 + 1
                diff9 = time_val.iat[i, 0][0 + j] - start_trace

            list_act.append([cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9])
            list_temp.append([diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8, diff9])
            j = j + 1
            cont = 0
            lenk = len(list_act) - 1
            while cont <= lenk:
                image[(max_trace - 1) - cont][0] = [list_act[lenk - cont][0], list_temp[lenk - cont][0]]
                image[(max_trace - 1) - cont][1] = [list_act[lenk - cont][1], list_temp[lenk - cont][1]]
                image[(max_trace - 1) - cont][2] = [list_act[lenk - cont][2], list_temp[lenk - cont][2]]
                image[(max_trace - 1) - cont][3] = [list_act[lenk - cont][3], list_temp[lenk - cont][3]]
                image[(max_trace - 1) - cont][4] = [list_act[lenk - cont][4], list_temp[lenk - cont][4]]
                image[(max_trace - 1) - cont][5] = [list_act[lenk - cont][5], list_temp[lenk - cont][5]]
                image[(max_trace - 1) - cont][6] = [list_act[lenk - cont][6], list_temp[lenk - cont][6]]
                image[(max_trace - 1) - cont][7] = [list_act[lenk - cont][7], list_temp[lenk - cont][7]]
                image[(max_trace - 1) - cont][8] = [list_act[lenk - cont][8], list_temp[lenk - cont][8]]
                cont = cont + 1
            if cont == 1:
                pass
            else:
                list_image.append(image)
                image = np.zeros(matrix_zero)
        i = i + 1
    return list_image

#import dataset
df, max_trace, n_caseid, n_activity = dataset_summary("dataset/helpdesk.csv")

#group by activity and timestamp by caseid
act = df.groupby('CaseID').agg({'Activity': lambda x: list(x)})
temp = df.groupby('CaseID').agg({'Timestamp': lambda x: list(x)})

#split dataset first 2/3 - 1/3
size = int((n_caseid/3)*2)

train_act = act[:size]
train_temp = temp[:size]

test_act = act[size:]
test_temp = temp[size:]

#generate training and test set
X_train = get_image(train_act, train_temp, max_trace, n_activity)
X_test = get_image(test_act, test_temp, max_trace, n_activity)

l_train = get_label(train_act)
l_test = get_label(test_act)

le = preprocessing.LabelEncoder()
l_train = le.fit_transform(l_train)
l_test = le.transform(l_test)
num_classes = le.classes_.size
print(list(le.classes_))

X_train = np.asarray(X_train)
l_train = np.asarray(l_train)

X_test = np.asarray(X_test)
l_test = np.asarray(l_test)

train_Y_one_hot = np_utils.to_categorical(l_train, num_classes)
test_Y_one_hot = np_utils.to_categorical(l_test, num_classes)

#define neural network architecture
model = Sequential()
reg = 0.0001
input_shape = (max_trace, n_activity, 2)
model.add(Conv2D(32, (2, 2), input_shape=input_shape, padding='same', kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(reg)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (4, 4), padding='same', kernel_regularizer=regularizers.l2(reg), ))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (8, 8), padding='same', kernel_regularizer=regularizers.l2(reg), ))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax', name='act_output'))

print(model.summary())

opt = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=opt, metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=6)
history = model.fit(X_train, {'act_output': train_Y_one_hot}, validation_split=0.2, verbose=1,
                    callbacks=[early_stopping], batch_size=128, epochs=500)
model.save("BPI12_W_complete.h5")

# Print confusion matrix for training data
y_pred_train = model.predict(X_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
print(classification_report(l_train, max_y_pred_train, digits=3))

score = model.evaluate(X_test, test_Y_one_hot, verbose=1)

print('\nAccuracy on test data: ', score[1])
print('\nLoss on test data: ', score[0])

y_pred_test = model.predict(X_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(test_Y_one_hot, axis=1)
print(classification_report(max_y_test, max_y_pred_test, digits=3))
