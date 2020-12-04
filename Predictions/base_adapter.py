from Utils.LogFile import LogFile
from keras import utils as ku
import time
import numpy as np
from keras.layers import Embedding, Dense, Input, Concatenate, Softmax, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint


def learn_model(log, attributes, epochs, early_stop):
    num_activities = len(log.values[log.activity]) + 1
    # Input + Embedding layer for every attribute
    input_layers = []
    embedding_layers = []
    for attr in attributes:
        if attr not in log.ignoreHistoryAttributes and attr != log.trace:
            for k in range(log.k):
                i = Input(shape=(1,), name=attr.replace(" ", "_").replace("(", "").replace(")","").replace(":","_") + "_Prev%i" % k)
                input_layers.append(i)
                e = Embedding(len(log.values[attr]) + 1, 32, embeddings_initializer="zeros")(i)
                embedding_layers.append(e)
    concat = Concatenate()(embedding_layers)

    # dense1 = Dense(32)(concat)
    drop = Dropout(0.2)(concat)
    dense2 = Dense(num_activities)(drop)

    flat = Flatten()(dense2)

    output = Softmax(name="output")(flat)

    model = Model(inputs=input_layers, outputs=[output])
    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999,
                epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    model.compile(loss={'output': 'categorical_crossentropy'},
                  optimizer=opt)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop)

    outfile = 'Models/model_{epoch:03d}-{val_loss:.2f}.h5'
    model_checkpoint = ModelCheckpoint(outfile,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto')

    x, y, vals = transform_data(log, [a for a in attributes if a != log.trace])

    model.fit(x=x, y=y,
              validation_split=0.2,
              verbose=2,
              callbacks=[early_stopping, model_checkpoint],
              batch_size=32,
              epochs=epochs)
    return model


def transform_data(log, columns):
    num_activities = len(log.values[log.activity]) + 1

    col_num_vals = {}
    for col in columns:
        if col == log.activity:
            col_num_vals[col] = num_activities
        else:
            col_num_vals[col] = log.contextdata[col].max() + 2

    inputs = []
    for _ in range(len(columns) * log.k - len(log.ignoreHistoryAttributes) * log.k):
        inputs.append([])
    outputs = []
    for row in log.contextdata.iterrows():
        row = row[1]
        i = 0
        for attr in columns:
            if attr not in log.ignoreHistoryAttributes:
                for k in range(log.k):
                    inputs[i].append(row[attr + "_Prev%i" % k])
                    i += 1
        outputs.append(row[log.activity])

    outputs = ku.to_categorical(outputs, num_activities)
    for i in range(len(inputs)):
        inputs[i] = np.array(inputs[i])
    return inputs, outputs, col_num_vals


def train(log, epochs, early_stop):
    return learn_model(log, log.attributes(), epochs, early_stop)


def test(log, model):
    inputs, expected, _ = transform_data(log, [a for a in log.attributes() if a != log.trace])
    predictions = model.predict(inputs)
    predict_vals = np.argmax(predictions, axis=1)
    expected_vals = np.argmax(expected, axis=1)

    return np.sum(np.equal(predict_vals, expected_vals)) / len(predict_vals)

def run_experiment(data, prefix_size, add_end_event, split_method, split_cases, train_percentage):
    logfile = LogFile(data, ",", 0, None, None, "case",
                      activity_attr="event", convert=False, k=prefix_size)
    if add_end_event:
        logfile.add_end_events()
    logfile.keep_attributes(["case", "event", "role"])
    logfile.convert2int()
    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(train_percentage, case=split_cases, method=split_method)

    with open("Baseline/results.txt", "a") as fout:
        fout.write("Data: " + data)
        fout.write("\nPrefix Size: " + str(prefix_size))
        fout.write("\nEnd event: " + str(add_end_event))
        fout.write("\nSplit method: " + split_method)
        fout.write("\nSplit cases: " + str(split_cases))
        fout.write("\nTrain percentage: " + str(train_percentage))
        fout.write("\nDate: " + time.strftime("%d.%m.%y-%H.%M", time.localtime()))
        fout.write("\n------------------------------------")

        baseline_acc = test(test_log, train(train_log, epochs=100, early_stop=10))
        fout.write("\nBaseline: " + str(baseline_acc))
        fout.write("\n")
        fout.write("====================================\n\n")

if __name__ == "__main__":
    run_experiment("../Data/BPIC15_1_sorted_new.csv", 10, False, "train-test", True, 70)
    # data = []
    # # data.append("../Data/Helpdesk.csv")
    # # data.append("../Data/BPIC15_1_sorted_new.csv")
    # # data.append("../Data/BPIC15_3_sorted_new.csv")
    # data.append("../Data/BPIC15_5_sorted_new.csv")
    # data.append("../Data/BPIC12W.csv")
    # data.append("../Data/BPIC12.csv")
    #
    # prefix_size = [1,2,4,5,10,15,20,25,30,35]
    # add_end_event = [True, False]
    # split_method = ["train-test", "test-train", "random"]
    # split_cases = [True, False]
    # train_percentage = [70, 80]
    #
    # for d in data:
    #     for ps in prefix_size:
    #         for aee in add_end_event:
    #             for sm in split_method:
    #                 for sc in split_cases:
    #                     for tp in train_percentage:
    #                         run_experiment(d, ps, aee, sm, sc, tp)