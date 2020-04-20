import copy
import os

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Nadam

from Utils.LogFile import LogFile

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


def train(log, epochs=500, early_stop=42):
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

    # build the model:
    print('Build model...')
    main_input = Input(shape=(maxlen, num_features), name='main_input')
    # train a 2-layer LSTM with one shared layer
    l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input) # the shared layer
    b1 = BatchNormalization()(l1)
    l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in activity prediction
    b2_1 = BatchNormalization()(l2_1)

    act_output = Dense(len(target_chars), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)

    model = Model(inputs=[main_input], outputs=[act_output])

    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

    model.compile(loss={'act_output':'categorical_crossentropy'}, optimizer=opt)
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop)
    model_checkpoint = ModelCheckpoint(os.path.join("model", 'model_{epoch:03d}-{val_loss:.2f}.h5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    model.fit(X, {'act_output':y_a}, validation_split=0.2, verbose=2, callbacks=[early_stopping, lr_reducer], batch_size=maxlen, epochs=epochs)

    return model


def test(log, model):
    lines = convert_log(log)

    maxlen = max(max(map(lambda x: len(x), lines)), model.input_shape[1])

    chars = [chr(i + ascii_offset) for i in range(len(log.values["event"]) + 1)]
    chars.sort()
    target_chars = copy.copy(chars)
    print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
    target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
    print(indices_char)

    # define helper functions
    def encode(sentence, maxlen=maxlen):
        num_features = len(chars)+1
        X = np.zeros((1, maxlen, num_features), dtype=np.float32)
        leftpad = maxlen-len(sentence)
        for t, char in enumerate(sentence):
            for c in chars:
                if c == char:
                    X[0, t+leftpad, char_indices[c]] = 1
            X[0, t+leftpad, len(chars)] = t+1
        return X

    def getSymbol(predictions):
        maxPrediction = 0
        symbol = ''
        i = 0
        for prediction in predictions:
            if(prediction>=maxPrediction):
                maxPrediction = prediction
                symbol = target_indices_char[i]
            i += 1
        return symbol

    results = {}
    all_results = []
    for prefix_size in range(1, maxlen):
        print(prefix_size)
        results[prefix_size] = []
        for line in lines:
            if prefix_size >= len(line):
                continue
            cropped_line = ''.join(line[:prefix_size])
            ground_truth = ''.join(line[prefix_size:prefix_size + 1])

            enc = encode(cropped_line)
            y = model.predict(enc, verbose=0)
            prediction = getSymbol(y[0])
            results[prefix_size].append(prediction == ground_truth)
            all_results.append(prediction == ground_truth)
    print("Accuracy:", sum(all_results) / len(all_results))
    return sum(all_results) / len(all_results)


if __name__ == "__main__":
    data = "../../Data/Camargo_Helpdesk.csv"
    # data = "../../Data/Taymouri_bpi_12_w.csv"
    case_attr = "case"
    act_attr = "event"

    logfile = LogFile(data, ",", 0, None, None, case_attr,
                      activity_attr=act_attr, convert=False, k=5)
    logfile.convert2int()

    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(80, case=True, method="random")

    model = train(train_log, epochs=100, early_stop=10)
    test(test_log, model)

