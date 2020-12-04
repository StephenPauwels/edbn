from datetime import datetime

import pandas as pd
import numpy as np

import multiprocessing as mp
from functools import partial

from Utils.LogFile import LogFile

seed = 123
np.random.seed(seed)


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
    return list_label


def dataset_summary(dataset):
    df = pd.read_csv(dataset, sep=",")
    print("Activity Distribution\n", df['event'].value_counts())
    n_caseid = df['case'].nunique()
    n_activity = df['Activity'].nunique()
    print("Number of CaseID", n_caseid)
    print("Number of Unique Activities", n_activity)
    print("Number of Activities", df['event'].count())
    cont_trace = df['case'].value_counts(dropna=False)
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
        conts = np.zeros(n_activity + 1)
        diffs = np.zeros(n_activity + 1)
        while j < (len(act_val.iat[i, 0]) - 1):
            start_trace = time_val.iat[i, 0][0]

            conts[act_val.iat[i, 0][0 + j]] += 1
            diffs[act_val.iat[i, 0][0 + j]] = time_val.iat[i, 0][0 + j] - start_trace

            list_act.append(conts[1:])
            list_temp.append(diffs[1:])
            j = j + 1
            cont = 0
            lenk = len(list_act) - 1
            while cont <= lenk:
                image[(max_trace - 1) - cont] = np.array(list(zip(list_act[lenk - cont], list_temp[lenk - cont])))

                cont = cont + 1
            if cont == 1:
                pass
            else:
                list_image.append(image)
                image = np.zeros(matrix_zero)
        i = i + 1
    return list_image


def get_image_from_log(log):
    n_activity = len(log.values[log.activity])
    matrix_zero = (log.k, n_activity, 2)
    list_image = []

    for row in log.contextdata.iterrows():
        image = np.zeros(matrix_zero)
        conts = np.zeros(n_activity + 1)
        diffs = np.zeros(n_activity + 1)
        starttime = None
        for i in range(log.k - 1, -1, -1):
            event = row[1]["%s_Prev%i" % (log.activity, i)]
            conts[event] += 1
            t_raw = row[1]["%s_Prev%i" % (log.time, i)]
            if t_raw != 0:
                try:
                    t = datetime.strptime(t_raw, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    t = datetime.strptime(t_raw, "%Y/%m/%d %H:%M:%S.%f")
                if starttime is None:
                    starttime = t
                diffs[event] = (t - starttime).total_seconds()
            image[log.k - 1 - i] = np.array(list(zip(conts[1:], diffs[1:])))
        list_image.append(image)
    return list_image

def get_image_from_log2(log):
    n_activity = len(log.values[log.activity])
    matrix_zero = (log.k, n_activity, 2)
    list_image = []

    create_image_func = partial(create_image, matrix_zero=matrix_zero, activity_attr=log.activity, time_attr=log.time)

    with mp.Pool(mp.cpu_count()) as p:
        list_image = p.map(create_image_func, log.contextdata.iterrows())

    return list_image

def create_image(row, matrix_zero, activity_attr, time_attr):
    image = np.zeros(matrix_zero)
    conts = np.zeros(matrix_zero[1] + 1)
    diffs = np.zeros(matrix_zero[1] + 1)
    starttime = None
    for i in range(matrix_zero[0] - 1, -1, -1):
        event = row[1]["%s_Prev%i" % (activity_attr, i)]
        conts[event] += 1
        t_raw = row[1]["%s_Prev%i" % (time_attr, i)]
        if t_raw != 0:
            try:
                t = datetime.strptime(t_raw, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                t = datetime.strptime(t_raw, "%Y/%m/%d %H:%M:%S.%f")
            if starttime is None:
                starttime = t
            diffs[event] = (t - starttime).total_seconds()
        image[matrix_zero[0] - 1 - i] = np.array(list(zip(conts[1:], diffs[1:])))
    return image

def get_label_from_log(log):
    list_label = []
    for row in log.contextdata.iterrows():
        list_label.append(row[1][log.activity])
    return list_label


def train(log, epochs=500, early_stop=42):
    from keras.models import Sequential
    from keras.layers.core import Flatten, Dense
    from keras.layers.convolutional import MaxPooling2D
    from keras.optimizers import Nadam
    from keras.callbacks import EarlyStopping
    from keras.layers.normalization import BatchNormalization
    from keras.layers import Conv2D, Activation
    from keras import regularizers
    from keras.utils import np_utils

    X_train = get_image_from_log(log)
    y_train = get_label_from_log(log)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    train_Y_one_hot = np_utils.to_categorical(y_train, len(log.values[log.activity]) + 1)

    trace_size = log.k
    n_activity = len(log.values[log.activity])

    #define neural network architecture
    model = Sequential()
    reg = 0.0001
    input_shape = (trace_size, n_activity, 2)
    model.add(Conv2D(32, (2, 2), input_shape=input_shape, padding='same', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (4, 4), padding='same', kernel_regularizer=regularizers.l2(reg), ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    if trace_size >= 8:
        model.add(Conv2D(128, (8, 8), padding='same', kernel_regularizer=regularizers.l2(reg), ))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(len(log.values[log.activity]) + 1, activation='softmax', name='act_output'))

    print(model.summary())

    opt = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=opt, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop)
    model.fit(X_train, {'act_output': train_Y_one_hot}, validation_split=0.2, verbose=1,
              callbacks=[early_stopping], batch_size=128, epochs=epochs)
    return model


def test(log, model):
    from keras.utils import np_utils

    X_test = get_image_from_log(log)
    y_test = get_label_from_log(log)

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    predictions = model.predict(X_test)

    predict_vals = np.argmax(predictions, axis=1)

    predict_probs = predictions[np.arange(predictions.shape[0]), predict_vals]
    result = zip(y_test, predict_vals, predict_probs)

    return result

if __name__ == "__main__":
    data = "../../Data/BPIC15_1_sorted_new.csv"
    case_attr = "case"
    act_attr = "event"

    logfile = LogFile(data, ",", 0, None, "completeTime", case_attr,
                      activity_attr=act_attr, convert=False, k=10)
    logfile.convert2int()

    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(80, case=True, method="train-test")

    model = train(train_log, epochs=100, early_stop=10)
    notest(test_log, model)