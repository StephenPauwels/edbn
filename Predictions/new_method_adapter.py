from EDBN.Execute import train as edbn_train
from LogFile import LogFile
import numpy as np

def transform_data(log, columns):
    from keras.utils import to_categorical

    num_activities = len(log.values[log.activity]) + 1

    col_num_vals = {}
    for col in columns:
        if col == log.activity:
            col_num_vals[col] = num_activities
        else:
            col_num_vals[col] = log.contextdata[col].max() + 2

    inputs = []
    for _ in range(len(columns)):
        inputs.append([])
    outputs = []
    for row in log.contextdata.iterrows():
        row = row[1]
        i = 0
        for attr in columns:
            inputs[i].append(row[str(attr)])
            i += 1
        outputs.append(row[log.activity])

    outputs = to_categorical(outputs, num_activities)
    for i in range(len(inputs)):
        inputs[i] = np.array(inputs[i])
    return inputs, outputs, col_num_vals


def learn_model(log, features, epochs, early_stop):
    from keras import Input, Model
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.layers import Embedding, Concatenate, Dropout, Dense, Flatten, Softmax
    from keras.optimizers import Nadam

    num_activities = len(log.values[log.activity]) + 1
    # Input + Embedding layer for every attribute
    input_layers = []
    embedding_layers = []
    for feature in features:
        i = Input(shape=(1,), name=feature.replace(" ", "_").replace("(", "").replace(")","").replace(":","_"))
        input_layers.append(i)
        e = Embedding(len(log.values[feature.rsplit("_", 1)[0]]) + 1, 32, embeddings_initializer="zeros")(i)
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

    x, y, vals = transform_data(log, features)

    model.fit(x=x, y=y,
              validation_split=0.2,
              verbose=2,
              callbacks=[early_stopping, model_checkpoint],
              batch_size=32,
              epochs=epochs)
    return model

def train(log, epochs, early_stop):
    # First train Bayesian network
    edbn_model = edbn_train(log)

    # Get all usefull features
    features = [var.attr_name for var in edbn_model.get_variable(log.activity).get_conditional_parents()]

    print("Using features:", features)

    return (learn_model(log, features, epochs, early_stop), features)


def test(log, model):
    nn_model = model[0]
    features = model[1]

    inputs, expected, _ = transform_data(log, features)
    predictions = nn_model.predict(inputs)
    predict_vals = np.argmax(predictions, axis=1)
    expected_vals = np.argmax(expected, axis=1)

    return np.sum(np.equal(predict_vals, expected_vals)) / len(predict_vals)

if __name__ == "__main__":
    # data = "../Data/Helpdesk.csv"
    data = "../Data/BPIC12W.csv"
    case_attr = "case"
    act_attr = "event"

    logfile = LogFile(data, ",", 0, None, None, case_attr,
                      activity_attr=act_attr, convert=False, k=5)

    prefix_size = max(logfile.data.groupby(logfile.trace).size())
    if prefix_size > 40:
        prefix_size = 40
    logfile.k = prefix_size

    logfile.add_end_events()

    logfile.keep_attributes(["case", "event", "role"])
    logfile.convert2int()

    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(66, case=False, method="train-test")

    model = train(train_log, 100, 10)
    acc = test(test_log, model)
    print(acc)