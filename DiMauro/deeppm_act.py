"""
@authors: Di Mauro, Appice and Basile
"""

import numpy as np
from keras.callbacks import ModelCheckpoint

seed = 123
np.random.seed(seed)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from DiMauro.utils import load_data, load_cases
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

    input = Input(shape=(input_length,))
    filters_inputs = Embedding(vocab_size, embedding_size, input_length=input_length)(input)

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
        model = Model(inputs=input, outputs=[out_a, out_t])
        model.compile(optimizer=optimizer, loss={'output_a':'categorical_crossentropy', 'output_t':'mae'})
    else:
        if (model_type=='ACT'):
            out = Dense(n_classes, activation='softmax')(pool)
            model = Model(inputs=input, outputs=out)
            model.compile(optimizer=optimizer, loss='mse',metrics=['acc'])
        elif (model_type=='TIME'):
            out = Dense(1, activation='linear')(pool)
            model = Model(inputs=input, outputs=out)
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
        h = model.fit(X_train,
                      y_train, epochs=200, verbose=2,
                      validation_split=0.2, callbacks=[early_stopping], batch_size=2**params['batch_size'])

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
    model_type = "ACT"
    output_file = os.path.join(model_folder, "output.log")

    global X_train, y_train

    (X_train, y_train,
     X_test, y_test,
     vocab_size,
     max_length,
     n_classes,
     prefix_sizes) = load_data(train_log, test_log, case_index=0, act_index=1)

    emb_size = (vocab_size + 1 ) // 2 # --> ceil(vocab_size/2)

    # categorical output
    y_train = to_categorical(y_train, num_classes=vocab_size)

    n_iter = 20

    space = {'input_length':max_length, 'vocab_size':vocab_size, 'n_classes':n_classes, 'model_type':model_type, 'embedding_size':emb_size,
             'n_modules':hp.choice('n_modules', [1,2,3]),
             'batch_size': hp.choice('batch_size', [4,5]),
             'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01))}

    # model selection
    print('Starting model selection...')
    global best_score, best_model, best_time, best_numparameters

    if params is None:
        current_time = time.strftime("%d.%m.%y-%H.%M", time.localtime())
        outfile = open(output_file, 'w')

        outfile.write("Starting time: %s\n" % current_time)

        print("No params given, starting parameter search")
        trials = Trials()
        best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=n_iter, trials=trials, rstate= np.random.RandomState(seed))
        best_params = hyperopt.space_eval(space, best)

        fit_and_score(best_params)

        outfile.write("\nHyperopt trials")
        outfile.write("\ntid,loss,learning_rate,n_modules,batch_size,time,n_epochs,n_params,perf_time")
        for trial in trials.trials:
            outfile.write("\n%d,%f,%f,%d,%d,%s,%d,%d,%f"%(trial['tid'],
                                                    trial['result']['loss'],
                                                    trial['misc']['vals']['learning_rate'][0],
                                                    int(trial['misc']['vals']['n_modules'][0]+1),
                                                    trial['misc']['vals']['batch_size'][0]+7,
                                                    (trial['refresh_time']-trial['book_time']).total_seconds(),
                                                    trial['result']['n_epochs'],
                                                    trial['result']['n_params'],
                                                    trial['result']['time']))

        outfile.write("\n\nBest parameters:")
        print(best_params, file=outfile)
        outfile.write("\nModel parameters: %d" % best_numparameters)
        outfile.write('\nBest Time taken: %f'%best_time)
        best_model.save(os.path.join(model_folder, "model.h5"))
    else:
        model = get_model(max_length, 3, vocab_size,
                          n_classes, emb_size, params["n_modules"], params["model_type"]
                          , params["learning_rate"])
        early_stopping = EarlyStopping(monitor='val_loss', patience=42)

        output_file_path = os.path.join(model_folder, 'model_rd_{epoch:03d}-{val_loss:.2f}.h5')

        # Saving
        model_checkpoint = ModelCheckpoint(output_file_path,
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='auto')

        model.fit(X_train, y_train, epochs=200, verbose=2,
                      validation_split=0.2, callbacks=[early_stopping, model_checkpoint], batch_size=2**params['batch_size'])

def evaluate(train_log, test_log, model_folder):
    print("MODEL", model_folder)
    model = load_model(model_folder)

    (X_train, y_train,
     X_test, y_test,
     vocab_size,
     max_length,
     n_classes,
     prefix_sizes) = load_data(train_log, test_log, case_index=0, act_index=1)

    # evaluate
    print('Evaluating final model...')
    preds_a = model.predict([X_test])

   # y_a_test = np.argmax(y_test, axis=1)
    preds_a = np.argmax(preds_a, axis=1)

    print(preds_a)
    print(y_test)

    accuracy = accuracy_score(y_test, preds_a)
    return accuracy

def predict_suffix(train_log, test_log, model_folder):
    model = load_model(model_folder)

    (train_cases,
     test_cases_X, test_cases_y,
     vocab_size,
     max_length,
     prefix_sizes) = load_cases(train_log, test_log, case_index=0, act_index=1)

    suffix_
    for test_x, test_y in zip(test_cases_X, test_cases_y):
        case = test_x
        suffix = []
        for _ in range(max_length):
            prediction = model.predict([[case]])
            pred_a = np.argmax(prediction, axis=1)
            suffix.append(pred_a[0])
            case = np.roll(case, -1)
            case[-1] = pred_a
            if pred_a[0] == -2:
                break
        similarities.append(1 - (damerau_levenshtein_distance(suffix, test_y)) / max(len(suffix), len(test_y)))
    return np.average(similarities)

def calc_suffix(input, max_length, model):
    test_x, test_y = input
    case = test_x
    suffix = []
    for _ in range(max_length):
        prediction = model.predict([[case]])
        pred_a = np.argmax(prediction, axis=1)
        suffix.append(pred_a[0])
        case = np.roll(case, -1)
        case[-1] = pred_a
        if pred_a[0] == -2:
            break
    return 1 - (damerau_levenshtein_distance(suffix, test_y)) / max(len(suffix), len(test_y))

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