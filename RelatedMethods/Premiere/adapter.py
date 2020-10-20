from Utils.LogFile import LogFile
import itertools
from dateutil.parser import parse
from itertools import tee
import Premiere.utility as ut
import numpy as np
import pandas as pd
from sklearn import preprocessing
import keras.utils as ku
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers import Input, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import keras


def train(log, epochs=200, early_stop=42):
    print("Start kometa_feature")
    kometa_feature = pd.DataFrame(generate_kometa_feature(log))
    print("Done kometa feature")
    X, num_col = generate_image(kometa_feature)

    y = []
    for case_id, case in log.get_cases():
        activities = list(case[log.activity])
        y.extend(activities[1:])

    y = ku.to_categorical(y, num_classes=len(log.values[log.activity]) + 1)

    X = np.asarray(X)

    return create_model(X, y, num_col, epochs, early_stop)


def test(log, model):
    kometa_feature = pd.DataFrame(generate_kometa_feature(log))
    X, num_col = generate_image(kometa_feature, True)

    y = []
    for case_id, case in log.get_cases():
        activities = list(case[log.activity])
        y.extend(activities[1:])

    X = np.asarray(X)

    preds_a = model.predict(X)

    preds_a = np.argmax(preds_a, axis=1)
    return sum(preds_a == y) / len(preds_a)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def generate_kometa_feature(log):
    list_sequence_prefix = []
    list_resource_prefix = []
    list_time_prefix = []

    target = []

    for case_id, case in log.get_cases():
        activities = list(case[log.activity])
        resources = list(case["role"])
        # Convert all dates to datetime.datetime object
        time = [parse(t) for t in list(case[log.time])]

        for i in range(1, len(activities)):
            list_sequence_prefix.append(activities[:i])
            list_resource_prefix.append(resources[:i])
            list_time_prefix.append(time[:i])
            target.append(activities[i])

    unique_events = len(log.values[log.activity])
    unique_resources = len(log.values['role'])

    listOfeventsInt = list(range(1, unique_events + 1))
    flow_act = [p for p in itertools.product(listOfeventsInt, repeat=2)]

    agg_time_feature = []
    i = 0
    while i < len(list_time_prefix):
        time_feature = []
        duration = list_time_prefix[i][-1] - list_time_prefix[i][0]
        time_feature.append((86400 * duration.days + duration.seconds) / 86400)
        time_feature.append(len(list_sequence_prefix[i]))
        if len(list_sequence_prefix[i]) == 1:
            time_feature.append(0)
            time_feature.append(0)
            time_feature.append(0)
            time_feature.append(0)
        else:
            diff_cons = [y - x for x, y in pairwise(list_time_prefix[i])]
            diff_cons_sec = [((86400 * item.days + item.seconds) / 86400) for item in diff_cons]
            time_feature.append(np.mean(diff_cons_sec))
            time_feature.append(np.median(diff_cons_sec))
            time_feature.append(np.min(diff_cons_sec))
            time_feature.append(np.max(diff_cons_sec))

        agg_time_feature.append(time_feature)
        i = i + 1

    return ut.premiere_feature(list_sequence_prefix, list_resource_prefix, flow_act, agg_time_feature, unique_events, unique_resources, target)


def dec_to_bin(x):
    return format(int(x), "b")


def flat_vec_parallel(df):
    print("Start flat_vec")
    import multiprocessing

    num_processes = multiprocessing.cpu_count()
    chunk_size = int(df.shape[0] / num_processes)

    chunks = [df.iloc[df.index[i:i + chunk_size]] for i in range(0, df.shape[0], chunk_size)]

    pool = multiprocessing.Pool(processes=num_processes)
    result = pool.map(flat_vec, chunks)
    list_image_flat = []
    for r in result:
        list_image_flat.extend(r)
    print("flat_vec done")
    return list_image_flat


def flat_vec(df):
    list_image_flat = []
    for (index_label, row_series) in df.iterrows():
        list_image = []
        j = 0
        while j < len(row_series):
            v = row_series[j] * (2 ** 24 - 1)
            bin_num = dec_to_bin(int(v))
            if len(bin_num) < 24:
                pad = 24 - len(bin_num)
                zero_pad = "0" * pad
                line = zero_pad + str(bin_num)
                n = 8
                rgb = [line[i:i + n] for i in range(0, len(line), n)]
            else:
                n = 8
                line = str(bin_num)
                rgb = [line[i:i + n] for i in range(0, len(line), n)]
            list_image.append(rgb)
            j = j + 1
        list_image_flat.append(list_image)
    return list_image_flat


def flat_vec_test(df):
    list_image_flat = []
    for (index_label, row_series) in df.iterrows():
        list_image = []
        j = 0
        while j < len(row_series):
            if row_series[j] > 1:
                c = 1.0
            elif row_series[j] < 0:
                c = 0.0
            else:
                c = row_series[j]

            v = c * (2 ** 24 - 1)
            bin_num = dec_to_bin(int(v))
            if len(bin_num) < 24:
                pad = 24 - len(bin_num)
                zero_pad = "0" * pad
                line = zero_pad + str(bin_num)
                n = 8
                rgb = [line[i:i + n] for i in range(0, len(line), n)]
            else:
                n = 8
                line = str(bin_num)
                rgb = [line[i:i + n] for i in range(0, len(line), n)]
            list_image.append(rgb)
            j = j + 1
        list_image_flat.append(list_image)
    return list_image_flat


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


def rgb_img(list_image_flat_train, num_col):
    x = 0
    list_rgb = []
    size, padding = get_image_size(num_col)
    vec = [[0, 0, 0]] * padding
    while x < len(list_image_flat_train):
        y = 0
        list_img = []
        while y < len(list_image_flat_train[x]):
            z = 0
            img = []
            while z < len(list_image_flat_train[x][y]):
                bin_num = (list_image_flat_train[x][y][z])
                int_num = int(bin_num, 2)
                img.append(int_num)
                z = z + 1
            list_img.append(img)
            y = y + 1
        list_img = list_img + vec
        new_img = np.asarray(list_img)
        new_img = new_img.reshape(size, size, 3)
        list_rgb.append(new_img)
        x = x + 1
    return list_rgb


def generate_image(df, test=False):
    num_col = len(df.columns) - 1
    X = df[:]
    X = X.iloc[:, :-1]

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X.values.astype(float))

    norm = scaler.transform(X.values.astype(float))
    norm = pd.DataFrame(norm)

    if test:
        list_image_flat = flat_vec_test(norm)
    else:
        list_image_flat = flat_vec_parallel(norm)

    return rgb_img(list_image_flat, num_col), num_col


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


def create_model(X, y, num_col, epochs, early_stop):
    seed = 123

    n_classes = len(y[0])

    dense1 = 64
    dense2 = 128
    dropout1 = 0.3
    dropout2 = 0.4
    learning_rate = 0.0002

    f1, f2, f3 = 64, 128, 32
    img_size, padding = get_image_size(num_col)

    X_a_train = X
    X_a_train = X_a_train.astype('float32')
    X_a_train = X_a_train / 255.0

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
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop)
    model_checkpoint = ModelCheckpoint("premiere_models/" + 'model_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    model.fit(X_a_train, y, epochs=epochs, batch_size=128, verbose=1, callbacks=[early_stopping, lr_reducer], validation_split=0.2)
    return model


if __name__ == "__main__":
    # data = "../../Data/Helpdesk.csv"
    data = "../../Data/BPIC15_1_sorted_new.csv"
    case_attr = "case"
    act_attr = "event"

    logfile = LogFile(data, ",", 0, None, "completeTime", case_attr,
                      activity_attr=act_attr, convert=False, k=40)
    logfile.convert2int()

    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(70, case=True, method="train-test")

    model = train(train_log,200,20)
    print("Accuracy:", test(test_log, model))

