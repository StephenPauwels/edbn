from Utils.LogFile import LogFile
from keras import utils as ku
import numpy as np
from keras.layers import Embedding, Dense, Input, Concatenate, Softmax, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint


def learn_model(log, attributes, epochs, early_stop):
    num_activities = len(log.values["event"]) + 1
    # Input + Embedding layer for every attribute
    input_layers = []
    embedding_layers = []
    for attr in attributes:
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

    x, y, vals = transform_data(log, attributes)

    model.fit(x=x, y=y,
              validation_split=0.2,
              verbose=2,
              callbacks=[early_stopping, model_checkpoint],
              batch_size=32,
              epochs=epochs)
    return model


def transform_data(log, columns):
    num_activities = len(log.values["event"]) + 1

    col_num_vals = {}
    for col in columns:
        if col == "event":
            col_num_vals[col] = num_activities
        else:
            col_num_vals[col] = log.contextdata[col].max() + 2

    inputs = []
    for _ in range(len(columns) * log.k):
        inputs.append([])
    outputs = []
    for row in log.contextdata.iterrows():
        row = row[1]
        i = 0
        for attr in columns:
            for k in range(log.k):
                inputs[i].append(row[attr + "_Prev%i" % k])
                i += 1
        outputs.append(row["event"])

    outputs = ku.to_categorical(outputs, num_activities)
    for i in range(len(inputs)):
        inputs[i] = np.array(inputs[i])
    return inputs, outputs, col_num_vals


def train(log, epochs, early_stop):
    return learn_model(log, ["event", "role"], epochs, early_stop)


def test(log, model):
    inputs, expected, _ = transform_data(log, ["event", "role"])
    predictions = model.predict(inputs)
    predict_vals = np.argmax(predictions, axis=1)
    expected_vals = np.argmax(expected, axis=1)

    return np.sum(np.equal(predict_vals, expected_vals)) / len(predict_vals)

if __name__ == "__main__":
    data = "../Data/Camargo_Helpdesk.csv"
    # data = "../../Data/Taymouri_bpi_12_w.csv"
    case_attr = "case"
    act_attr = "event"

    logfile = LogFile(data, ",", 0, None, None, case_attr,
                      activity_attr=act_attr, convert=False, k=5)
    logfile.convert2int()

    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(80, case=True, method="train-test")

    model = train(train_log, epochs=100, early_stop=5)
    acc = test(test_log, model)
    print(acc)