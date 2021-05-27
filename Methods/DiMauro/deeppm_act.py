"""
@authors: Di Mauro, Appice and Basile
"""

import numpy as np
from keras.callbacks import ModelCheckpoint

seed = 123
np.random.seed(seed)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from Methods.DiMauro.utils import load_data_new, load_cases_new
from sklearn.metrics import accuracy_score

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from time import perf_counter
import time
import os

from tensorflow.keras.utils import Sequence

best_score = np.inf
best_model = None
best_time = 0
best_numparameters = 0

X_train = None
y_train = None

class DataGenerator(Sequence):
    def __init__(self, features, labels, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.labels = labels
        self.X_a = features[0]
        self.X_t = features[1]
        self.y_a = labels[0]
        self.y_t = labels[1]
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of steps per epoch'
        return int(np.floor(self.X_a.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.X_a.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_a = np.empty((self.batch_size, self.X_a.shape[1]))
        X_t = np.empty((self.batch_size, self.X_t.shape[1]))
        y_a = np.empty((self.batch_size, self.y_a.shape[1]), dtype=int)
        y_t = np.empty((self.batch_size))
           
                       

        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample
            X_a[i] = self.X_a[ID]
            X_t[i] = self.X_t[ID]

            # Store class
            y_a[i] = self.y_a[ID]
            y_t[i] = self.y_t[ID]
                       

        return [X_a, X_t], {'output_a':y_a, 'output_t':y_t}
    

def get_model(input_length=10, n_filters=3, vocab_size=10, n_classes=9, embedding_size=5, n_modules=5, model_type='ACT', learning_rate=0.002):
    #inception model

    inputs = []
    for i in range(2):
        inputs.append(Input(shape=(input_length,)))

    inputs_ = []
    for i in range(2):
        if (i==0):
            a = Embedding(vocab_size, embedding_size, input_length=input_length)(inputs[0])
            inputs_.append(Embedding(vocab_size, embedding_size, input_length=input_length)(inputs[i]))
        else:
            inputs_.append(Reshape((input_length, 1))(inputs[i]))

    filters_inputs = Concatenate(axis=2)(inputs_)

    for m in range(n_modules):
        filters = []
        for i in range(n_filters):
            filters.append(Conv1D(filters=32, strides=1, kernel_size=1+i, activation='relu', padding='same')(filters_inputs))
        filters.append(MaxPooling1D(pool_size=3, strides=1, padding='same')(filters_inputs))
        filters_inputs = Concatenate(axis=2)(filters)
        #filters_inputs = Dropout(0.1)(filters_inputs)

    #pool = GlobalAveragePooling1D()(filters_inputs)
    pool = GlobalMaxPooling1D()(filters_inputs)
    #pool = Flatten()(filters_inputs)

    #pool = Dense(64, activation='relu')(pool)


    optimizer = Adam(lr=learning_rate)

    if (model_type == 'BOTH'):
        out_a = Dense(n_classes, activation='softmax', name='output_a')(pool)
        out_t = Dense(1, activation='linear', name='output_t')(pool)
        model = Model(inputs=inputs, outputs=[out_a, out_t])
        model.compile(optimizer=optimizer, loss={'output_a':'categorical_crossentropy', 'output_t':'mae'})
    else:
        if (model_type=='ACT'):
            out = Dense(n_classes, activation='softmax')(pool)
            model = Model(inputs=inputs, outputs=out)
            model.compile(optimizer=optimizer, loss='mse',metrics=['acc'])
        elif (model_type=='TIME'):
            out = Dense(1, activation='linear')(pool)
            model = Model(inputs=inputs, outputs=out)
            model.compile(optimizer=optimizer, loss='mae')

    model.summary()

    return model


def fit_and_score(params):
    print(params)
    start_time = perf_counter()

    model = get_model(input_length=params['input_length'], vocab_size=params['vocab_size'], n_classes=params['n_classes'], model_type=params['model_type'],
                      learning_rate=params['learning_rate'], embedding_size=params['embedding_size'], n_modules=params['n_modules'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
   
    if (params['model_type'] == 'ACT'):
        # h = model.fit(params["X_train"],
        #               params["Y_train"], epochs=200, verbose=2,
        #               validation_split=0.2, callbacks=[early_stopping], batch_size=2**params['batch_size'])
        if len(params["X"]) < 10:
            split = 0
        else:
            split = 0.2
        h = model.fit([params['X'], params['X_t']], params['y'], epochs=params["epochs"], verbose=2,
                      validation_split=split, callbacks=[early_stopping], batch_size=2**params['batch_size'])

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

    return {'loss': score, 'status': STATUS_OK,  'n_epochs':  len(h.history['loss']), 'n_params':model.count_params(), 'time':end_time - start_time}


def train(train_log, test_log, model_folder, params):
    X, X_t, y = load_data_new(train_log)

    emb_size = (len(train_log.values["event"]) + 1) // 2  # --> ceil(vocab_size/2)
    y = to_categorical(y, num_classes=len(train_log.values["event"]) + 1)

    if params is None:
        n_iter = 3

        space = {'input_length': train_log.k, 'vocab_size': len(train_log.values["event"]) + 1,
                 'n_classes': len(train_log.values["event"]) + 1, 'model_type': "ACT",
                 'embedding_size': emb_size,
                 'n_modules': hp.choice('n_modules', [1, 2, 3]),
                 'batch_size': 5,
                 'learning_rate': 0.002, 'X': X, 'X_t': X_t, 'y': y,
                 'epochs': 200}

        trials = Trials()
        best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=n_iter, trials=trials,
                    rstate=np.random.RandomState(123))
        params = hyperopt.space_eval(space, best)

    model = get_model(train_log.k, 3, len(train_log.values["event"]) + 1,
                      len(train_log.values["event"]) + 1, emb_size, params["n_modules"], "ACT", 0.002)
    early_stopping = EarlyStopping(monitor='val_loss', patience=42)

    output_file_path = os.path.join("tmp", 'model_{epoch:03d}-{val_loss:.2f}.h5')

    # Saving
    model_checkpoint = ModelCheckpoint(output_file_path,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto')

    if len(y) < 10:
        split = 0
    else:
        split = 0.2
    model.fit([X, X_t], y, epochs=200, verbose=2, validation_split=split, callbacks=[early_stopping],
              batch_size=2 ** 5)
    model.save(os.path.join(model_folder, "model.h5"))


def evaluate(train_log, test_log, model_folder):
    print("MODEL", model_folder)
    model = load_model(model_folder)

    X, X_t, y = load_data_new(test_log)


    # evaluate
    print('Evaluating final model...')
    preds_a = model.predict([X, X_t])

    predict_vals = np.argmax(preds_a, axis=1)
    predict_probs = preds_a[np.arange(preds_a.shape[0]), predict_vals]
    expect_probs = preds_a[np.arange(preds_a.shape[0]), y]
    result = zip(y, predict_vals, predict_probs, expect_probs)
    return sum([1 if a[0] == a[1] else 0 for a in result]) / len(predict_vals)


def predict_suffix(model, data):
    X, X_t, y = load_cases_new(data.test_orig)

    suffix = [np.array([]) for _ in range(len(y))]

    length = 0
    while length < 100:
        predictions = model.predict([X, X_t])
        pred_a = np.argmax(predictions, axis=1)
        X = np.roll(X, -1, axis=1)
        X[:, -1] = pred_a
        suffix = np.concatenate((suffix, [[i] for i in pred_a]), axis=1)
        length += 1

    results = []
    for i, suf in enumerate(suffix):
        if 0 in suf:
            suf = suf[:np.where(suf == 0)[0][0]]
        results.append(1 - (damerau_levenshtein_distance(suf, y[i])) / max(len(suf), len(y[i])))
    return np.average(results)


"""
Compute the Damerau-Levenshtein distance between two given
lists (s1 and s2)
From: https://www.guyrutenberg.com/2008/12/15/damerau-levenshtein-distance-in-python/
"""
def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]