import os
from datetime import datetime

import hyperopt
import numpy as np
from hyperopt import Trials, tpe, fmin, hp
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from RelatedMethods.DiMauro.deeppm_act import fit_and_score, get_model
from Utils.LogFile import LogFile


def train(log, epochs=500, early_stop=4, params=None):
    X, X_t, y = load_data(log)

    emb_size = (len(log.values["event"]) + 1) // 2  # --> ceil(vocab_size/2)
    y = to_categorical(y, num_classes=len(log.values["event"]) + 1)

    if params is None:
        n_iter = 3

        space = {'input_length': log.k, 'vocab_size': len(log.values["event"]) + 1,
                 'n_classes': len(log.values["event"]) + 1, 'model_type': "ACT",
                 'embedding_size': emb_size,
                 'n_modules': hp.choice('n_modules', [1, 2, 3]),
                 'batch_size': 5,
                 'learning_rate': 0.002, 'X': X, 'X_t': X_t, 'y': y}

        trials = Trials()
        best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=n_iter, trials=trials,
                    rstate=np.random.RandomState(123))
        params = hyperopt.space_eval(space, best)

    model = get_model(log.k, 3, len(log.values["event"]) + 1,
                      len(log.values["event"]) + 1, emb_size, params["n_modules"], "ACT", 0.002)
    early_stopping = EarlyStopping(monitor='val_loss', patience=42)

    output_file_path = os.path.join("tmp", 'model_{epoch:03d}-{val_loss:.2f}.h5')

    # Saving
    model_checkpoint = ModelCheckpoint(output_file_path,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto')

    model.fit([X, X_t], y, epochs=200, verbose=2, validation_split=0.2, callbacks=[early_stopping, model_checkpoint],
              batch_size=2 ** 5)
    return model


def load_data(log):
    X = []
    X_t = []
    y = []

    casestarttime = None
    lasteventtime = None

    for case in log.get_cases():
        case_df = case[1]
        for row in case_df.iterrows():
            row = row[1]
            t_raw = row[log.time + "_Prev%i" % (log.k-1)]
            if t_raw != 0:
                try:
                    t = datetime.strptime(t_raw, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    t = datetime.strptime(t_raw, "%Y/%m/%d %H:%M:%S.%f")
                lasteventtime = t
            line = []
            times = []
            for i in range(log.k - 1, -1, -1):
                line.append(row["event_Prev%i" % i])
                t_raw = row[log.time + "_Prev%i" % i]
                if t_raw != 0:
                    try:
                        t = datetime.strptime(t_raw, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        t = datetime.strptime(t_raw, "%Y/%m/%d %H:%M:%S.%f")
                    if lasteventtime is None:
                        times.append(1)
                    else:
                        timesincelastevent = t - lasteventtime
                        timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds + timesincelastevent.microseconds/1000000
                        if timediff + 1 <= 0:
                            times.append(1)
                        else:
                            times.append(timediff+1)
                    lasteventtime = t
                else:
                    times.append(1) #to avoid zero
            X.append(line)
            X_t.append(times)
            y.append(row["event"])

    X = np.array(X)
    X_t = np.array(X_t)
    y = np.array(y)

    X_t = np.log(X_t)

    print("X:", X)
    print("X_t:", X_t)
    print("y:", y)

    return X, X_t, y


def test(log, model):
    X, X_t, y = load_data(log)

    # evaluate
    print('Evaluating final model...')
    preds_a = model.predict([X, X_t])

    # y_a_test = np.argmax(y_test, axis=1)
    preds_a = np.argmax(preds_a, axis=1)

    print(preds_a)
    print(y)

    accuracy = accuracy_score(y, preds_a)
    return accuracy


if __name__ == "__main__":
    # data = "../../Data/Helpdesk.csv"
    data = "../../Data/BPIC12W.csv"
    case_attr = "case"
    act_attr = "event"

    logfile = LogFile(data, ",", 0, None, time_attr="completeTime", trace_attr=case_attr,
                      activity_attr=act_attr, convert=False, k=5)
    logfile.convert2int()

    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(80, case=True, method="random")

    model = train(train_log, epochs=100, early_stop=10)
    acc = test(test_log, model)
    print(acc)