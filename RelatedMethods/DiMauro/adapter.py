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
                 'learning_rate': 0.002, 'X': X, 'X_t': X_t, 'y': y,
                 'epochs': epochs}

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

    if len(y) < 10:
        split = 0
    else:
        split = 0.2
    model.fit([X, X_t], y, epochs=epochs, verbose=2, validation_split=split, callbacks=[early_stopping],
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

    return X, X_t, y


def update(model, log):
    X, X_t, y = load_data(log)
    y = to_categorical(y, num_classes=len(log.values["event"]) + 1)

    if len(X) < 10:
        split = 0
    else:
        split = 0.2

    model.fit([X, X_t], y, epochs=10, verbose=0, validation_split=split,
              batch_size=log.k)


def test(model, log):
    X, X_t, y = load_data(log)

    # evaluate
    preds_a = model.predict([X, X_t])

    # y_a_test = np.argmax(y_test, axis=1)
    # preds_a = np.argmax(preds_a, axis=1)

    # accuracy = accuracy_score(y, preds_a)
    predict_vals = np.argmax(preds_a, axis=1)
    predict_probs = preds_a[np.arange(preds_a.shape[0]), predict_vals]
    result = zip(y, predict_vals, predict_probs)

    return result


def test_and_update(logs, model):
    results = []
    i = 0
    for t in logs:
        print(i, "/", len(logs))
        i += 1
        log = logs[t]["data"]
        results.extend(test(log, model))

        update(log, model, 10)

    return results

def test_and_update_retain(test_logs, model, train_log):
    results = []
    i = 0
    X_train, X_t_train, y_train = load_data(train_log)
    y_train = to_categorical(y_train, num_classes=len(train_log.values["event"]) + 1)
    for t in test_logs:
        print(i, "/", len(test_logs))
        i += 1
        test_log = test_logs[t]["data"]
        results.extend(test(test_log, model))

        X, X_t, y = load_data(test_log)
        y = to_categorical(y, num_classes=len(train_log.values["event"]) + 1)
        X_train = np.concatenate((X_train, X))
        X_t_train = np.concatenate((X_t_train, X_t))
        y_train = np.concatenate((y_train, y))
        model.fit([X_train, X_t_train], y_train, epochs=1, verbose=1, validation_split=0.2,
                  batch_size=train_log.k)

    return results


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