from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dropout, Concatenate, LSTM, Dense, BatchNormalization
from keras.optimizers import Nadam
import keras.utils as ku
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
from nltk.util import ngrams
import os

from Lin.Modulator import Modulator

def create_model(vec, vocab_act_size, vocab_role_size, output_folder):
    # Create embeddings + Concat
    act_input = Input(shape = (vec['prefixes']['x_ac_inp'].shape[1],), name="act_input")
    role_input = Input(shape = (vec['prefixes']['x_rl_inp'].shape[1],), name="role_input")

    act_embedding = Embedding(vocab_act_size, 100, input_length=vec['prefixes']['x_ac_inp'].shape[1],)(act_input)
    act_dropout = Dropout(0.2)(act_embedding)
    act_e_lstm_1 = LSTM(32, return_sequences=True)(act_dropout)
    act_e_lstm_2 = LSTM(100, return_sequences=True)(act_e_lstm_1)


    role_embedding = Embedding(vocab_role_size, 100, input_length=vec['prefixes']['x_rl_inp'].shape[1],)(role_input)
    role_dropout = Dropout(0.2)(role_embedding)
    role_e_lstm_1 = LSTM(32, return_sequences=True)(role_dropout)
    role_e_lstm_2 = LSTM(100, return_sequences=True)(role_e_lstm_1)


    concat1 = Concatenate(axis=1)([act_e_lstm_2, role_e_lstm_2])
    normal = BatchNormalization()(concat1)

    act_modulator = Modulator(attr_idx=0, num_attrs=1)(normal)
    role_modulator = Modulator(attr_idx=1, num_attrs=1)(normal)



    # Use LSTM to decode events
    act_d_lstm_1 = LSTM(100, return_sequences=True)(act_modulator)
    act_d_lstm_2 = LSTM(32, return_sequences=False)(act_d_lstm_1)

    act_output = Dense(vocab_act_size, name="act_output", activation='softmax')(act_d_lstm_2)

    model = Model(inputs=[act_input, role_input], outputs=[act_output])

    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999,
                epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=opt)

    model.summary()

    output_file_path = os.path.join(output_folder, 'model_rd_{epoch:02d}-{val_loss:.2f}.h5')

    # Saving
    model_checkpoint = ModelCheckpoint(output_file_path,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    model.fit({'act_input':vec['prefixes']['x_ac_inp'],
               'role_input':vec['prefixes']['x_rl_inp']},
              {'act_output':vec['next_evt']['y_ac_inp']},
              validation_split=0.2,
              verbose=2,
              batch_size=5,
              callbacks=[early_stopping, model_checkpoint],
              epochs=200)


def predict_next(model_file, df_test, case_attr="case", activity_attr="event"):
    model = load_model(os.path.join("models", model_file), custom_objects={'Modulator':Modulator})

    prefixes = create_pref_suf(df_test, case_attr, activity_attr)
    prefixes = predict(model, prefixes)

    accuracy = (np.sum([x['ac_true'] for x in prefixes]) / len(prefixes))

    print("Accuracy:", accuracy)


def vectorization(log_df, case_attr, activity="event", role="role", num_classes=None):
    """Example function with types documented in the docstring.
    Args:
        log_df (dataframe): event log data.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        args (dict): parameters for training the network
    Returns:
        dict: Dictionary that contains all the LSTM inputs.
    """
    cases = log_df[case_attr].unique()
    train_df = []
    for case in cases:
        train_df.append(log_df[log_df[case_attr] == case])


    vec = {'prefixes':dict(), 'next_evt':dict()}
    # n-gram definition
    for i, _ in enumerate(train_df):
        ac_n_grams = list(ngrams(train_df[i][activity], 5,
                                 pad_left=True, left_pad_symbol=0))
        rl_n_grams = list(ngrams(train_df[i][role], 5,
                                 pad_left=True, left_pad_symbol=0))

        st_idx = 0
        if i == 0:
            vec['prefixes']['x_ac_inp'] = np.array([ac_n_grams[0]])
            vec['prefixes']['x_rl_inp'] = np.array([rl_n_grams[0]])
            vec['next_evt']['y_ac_inp'] = np.array(ac_n_grams[1][-1])
            vec['next_evt']['y_rl_inp'] = np.array(rl_n_grams[1][-1])
            st_idx = 1
        for j in range(st_idx, len(ac_n_grams)-1):
            vec['prefixes']['x_ac_inp'] = np.concatenate((vec['prefixes']['x_ac_inp'],
                                                          np.array([ac_n_grams[j]])), axis=0)
            vec['prefixes']['x_rl_inp'] = np.concatenate((vec['prefixes']['x_rl_inp'],
                                                          np.array([rl_n_grams[j]])), axis=0)
            vec['next_evt']['y_ac_inp'] = np.append(vec['next_evt']['y_ac_inp'],
                                                    np.array(ac_n_grams[j+1][-1]))
            vec['next_evt']['y_rl_inp'] = np.append(vec['next_evt']['y_rl_inp'],
                                                    np.array(rl_n_grams[j+1][-1]))

    vec['next_evt']['y_ac_inp'] = ku.to_categorical(vec['next_evt']['y_ac_inp'], num_classes=num_classes)
    vec['next_evt']['y_rl_inp'] = ku.to_categorical(vec['next_evt']['y_rl_inp'])
    return vec


def create_pref_suf(df_test, case_attr="case", activity_attr="event"):
    """Extraction of prefixes and expected suffixes from event log.
    Args:
        df_test (dataframe): testing dataframe in pandas format.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        pref_size (int): size of the prefixes to extract.
    Returns:
        list: list of prefixes and expected sufixes.
    """
    prefixes = list()
    cases = df_test[case_attr].unique()
    for case in cases:
        trace = df_test[df_test[case_attr] == case]
        ac_pref = list()
        rl_pref = list()
        t_pref = list()
        for i in range(0, len(trace)-1):
            ac_pref.append(trace.iloc[i][activity_attr])
            rl_pref.append(trace.iloc[i]['role'])
            prefixes.append(dict(ac_pref=ac_pref.copy(),
                                 ac_next=trace.iloc[i + 1][activity_attr],
                                 rl_pref=rl_pref.copy(),
                                 rl_next=trace.iloc[i + 1]['role'],
                                 t_pref=t_pref.copy()))
    return prefixes


def predict(model, prefixes):
    """Generate business process suffixes using a keras trained model.
    Args:
        model (keras model): keras trained model.
        prefixes (list): list of prefixes.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        imp (str): method of next event selection.
    """
    # Generation of predictions
    for prefix in prefixes:

        # Activities and roles input shape(1,5)
        x_ac_ngram = np.append(
            np.zeros(5),
            np.array(prefix['ac_pref']),
            axis=0)[-5:].reshape((1, 5))

        x_rl_ngram = np.append(
            np.zeros(5),
            np.array(prefix['rl_pref']),
            axis=0)[-5:].reshape((1, 5))

        predictions = model.predict([x_ac_ngram, x_rl_ngram])

        pos = np.argmax(predictions[0])

        # Activities accuracy evaluation
        if pos == prefix['ac_next']:
            prefix['ac_true'] = 1
        else:
            prefix['ac_true'] = 0

    return prefixes

def train(logfile, train_log, model_folder):
    create_model(vectorization(train_log.data, train_log.trace, "event", num_classes=len(logfile.values[logfile.activity]) + 1), len(logfile.values[logfile.activity]) + 1, len(logfile.values["role"]) + 1, model_folder)



if __name__ == "__main__":
    import Preprocessing as data

    dataset = data.BPIC15_5
    logfile, name = data.get_data(dataset, 20000000, 0, False, False, False, False)
    train_log, test_log = logfile.splitTrainTest(70)

    #create_model(vectorization(train_log.data, train_log.trace, "Activity", num_classes=len(logfile.values[logfile.activity]) + 1), len(logfile.values[logfile.activity]) + 1, len(logfile.values["role"]) + 1, dataset)
    predict_next("BPIC15_5/model_rd_22-1.52.h5", test_log.data, test_log.trace, test_log.activity)
