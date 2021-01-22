import copy
import os

import numpy as np

from Utils.LogFile import LogFile

import tensorflow as tf

ascii_offset = 161

def convert_log(log):
    prefix_size = log.k
    lines = []  # these are all the activity seq
    for name, group in log.contextdata.groupby(log.trace):
        for row in group.iterrows():
            row = row[1]
            line = ""
            for i in range(prefix_size - 1, -1, -1):
                line += chr(int(row["event_Prev%i" % i] + ascii_offset))
            line += chr(int(row["event"] + ascii_offset))
            lines.append(line)
    return lines


def transform_log(log):
    activities = log.values[log.activity]
    X = np.zeros((len(log.contextdata), log.k, len(activities) + 1), dtype=np.float32)
    y_a = np.zeros((len(log.contextdata), len(activities) + 1), dtype=np.float32)
    j = 0
    for row in log.contextdata.iterrows():
        act = getattr(row[1], log.activity)
        k = 0
        for i in range(log.k -1, -1, -1):
            X[j, log.k - i - 1, getattr(row[1], "%s_Prev%i" % (log.activity, i))] = 1
            X[j, log.k - i - 1, len(activities)] = k
            k += 1
        y_a[j, act] = 1
        j += 1
    return X, y_a


def train(log, epochs=10, early_stop=42):
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from keras.layers import Input
    from keras.layers.core import Dense
    from keras.layers.normalization import BatchNormalization
    from keras.layers.recurrent import LSTM
    from keras.models import Model
    from keras.optimizers import Nadam
    """
    transform_log(log)
    input("Waiting")
    
    # lines = convert_log(log)
    lines = convert_log(log)

    maxlen = max(map(lambda x: len(x), lines)) #find maximum line size

    # next lines here to get all possible characters for events and annotate them with numbers
    chars = [chr(i + ascii_offset) for i in range(len(log.values["event"]) + 1)]
    chars.sort()
    target_chars = copy.copy(chars)
    print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
    target_indices_char = dict((i, c) for i, c in enumerate(target_chars))

    sentences = []
    next_chars = []

    for line in lines:
        sentences.append(line[:-1])
        next_chars.append(line[-1])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    num_features = len(chars)+1
    print('num features: {}'.format(num_features))
    X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
    y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        leftpad = maxlen-len(sentence)
        for t, char in enumerate(sentence):
            for c in chars:
                if c == char: #this will encode present events to the right places
                    X[i, t+leftpad, char_indices[c]] = 1
                    break
            X[i, t+leftpad, len(chars)] = t+1
        for c in target_chars:
            if c == next_chars[i]:
                y_a[i, target_char_indices[c]] = 1
            else:
                y_a[i, target_char_indices[c]] = 0
        #np.set_printoptions(threshold=np.nan)
    """
    X, y_a = transform_log(log)

    # build the model:
    print('Build model...')
    main_input = Input(shape=(log.k, len(log.values[log.activity])+1), name='main_input')
    # train a 2-layer LSTM with one shared layer
    l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input) # the shared layer
    b1 = BatchNormalization()(l1)
    l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in activity prediction
    b2_1 = BatchNormalization()(l2_1)

    act_output = Dense(len(log.values[log.activity]) + 1, activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)

    model = Model(inputs=[main_input], outputs=[act_output])

    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

    model.compile(loss={'act_output':'categorical_crossentropy'}, optimizer=opt)
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop)
    model_checkpoint = ModelCheckpoint(os.path.join("model", 'model_{epoch:03d}-{val_loss:.2f}.h5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    model.fit(X, {'act_output': y_a}, validation_split=0.2, verbose=2, callbacks=[early_stopping, lr_reducer], batch_size=log.k, epochs=epochs)

    return model


def test(log, model):
    X, y = transform_log(log)
    predictions = model.predict(X)
    predict_vals = np.argmax(predictions, axis=1)
    predict_probs = predictions[np.arange(predictions.shape[0]), predict_vals]
    expected_vals = np.argmax(y, axis=1)
    result = zip(expected_vals, predict_vals, predict_probs)
    return result

    # lines = convert_log(log)
    #
    # maxlen = max(max(map(lambda x: len(x), lines)), model.input_shape[1])
    #
    # chars = [chr(i + ascii_offset) for i in range(len(log.values["event"]) + 1)]
    # chars.sort()
    # target_chars = copy.copy(chars)
    # print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
    # char_indices = dict((c, i) for i, c in enumerate(chars))
    # indices_char = dict((i, c) for i, c in enumerate(chars))
    # target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
    # target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
    # print(indices_char)
    #
    # # define helper functions
    # def encode(sentence, maxlen=maxlen):
    #     num_features = len(chars)+1
    #     X = np.zeros((1, maxlen, num_features), dtype=np.float32)
    #     leftpad = maxlen-len(sentence)
    #     for t, char in enumerate(sentence):
    #         for c in chars:
    #             if c == char:
    #                 X[0, t+leftpad, char_indices[c]] = 1
    #         X[0, t+leftpad, len(chars)] = t+1
    #     return X
    #
    # results = {}
    # all_results = []
    # for prefix_size in range(1, maxlen):
    #     print(prefix_size)
    #     results[prefix_size] = []
    #     for line in lines:
    #         if prefix_size >= len(line):
    #             continue
    #         cropped_line = ''.join(line[:prefix_size])
    #         ground_truth = ''.join(line[prefix_size:prefix_size + 1])
    #         if ground_truth == "¡":
    #             continue
    #
    #         enc = encode(cropped_line)
    #         y = model.predict(enc, verbose=0)
    #
    #         predicted_val = np.argmax(y[0])
    #         predicted_prob = y[0][predicted_val]
    #         all_results.append((ground_truth, target_indices_char[predicted_val], predicted_prob))
    #
    # return all_results


def test_and_update(logs, model):
    results = []
    i = 0
    for t in logs:
        print(i, "/", len(logs))
        i += 1
        log = logs[t]["data"]
        results.extend(test(log, model))

        X, y = transform_log(log)
        if len(X) < 10:
            split = 0
        else:
            split = 0.2
        model.fit(X, {'act_output': y}, validation_split=split, verbose=0,
                  batch_size=log.k, epochs=1)

    return results


def test_and_update_retain(test_logs, model, train_log):
    import gc

    train_x, train_y = transform_log(train_log)

    results = []
    i = 0
    for t in test_logs:
        print(i, "/", len(test_logs))
        i += 1
        test_log = test_logs[t]["data"]
        results.extend(test(test_log, model))
        test_x, test_y = transform_log(test_log)
        train_x = np.concatenate((train_x, test_x))
        train_y = np.concatenate((train_y, test_y))
        model.fit(train_x, {'act_output': train_y},
                  validation_split=0.2,
                  verbose=0,
                  batch_size=train_log.k,
                  epochs=1)
        gc.collect() # Avoid that too many batches lead to full memory
    return results

if __name__ == "__main__":
    data = "../../Data/BPIC12W.csv"
    # data = "../../Data/Helpdesk.csv"
    # data = "../../Data/Taymouri_bpi_12_w.csv"
    case_attr = "case"
    act_attr = "event"
    k = 15

    logfile = LogFile(data, ",", 0, None, None, case_attr,
                      activity_attr=act_attr, convert=False, k=10)
    logfile.convert2int()
    logfile.filter_case_length(5)
    # logfile.k = min(k, max(logfile.data.groupby(logfile.trace).size()))

    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(70, case=False, method="test-train")

    model = train(train_log, epochs=100, early_stop=10)
    # model.save("tmp.h5")
    # from keras.models import load_model
    # test(test_log, load_model("tmp.h5"))
    test(test_log, model)
