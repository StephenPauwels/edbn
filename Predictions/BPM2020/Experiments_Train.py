"""
    Basic file to train the models for the prediction experiments
    Use: python Experiments_Train METHOD DATA [args]

    Valid values for METHOD and DATA can be found in Experiments_Variables.py

    Extra arguments:
        - EDBN: * first extra argument: k-value
                * second extra argument: use "next" when training optimized for next event, use "suffix" for suffix
        - CAMARGO: specify architecture to use: chared_cat or specialized

    Author: Stephen Pauwels
"""

import os
import pickle
import sys
import time

from RelatedMethods.Camargo.support_modules.support import create_csv_file_header
from BPM2020.Experiments_Variables import DATA_DESC, DATA_FOLDER, OUTPUT_FOLDER
from BPM2020.Experiments_Variables import EDBN, CAMARGO, DIMAURO, LIN, TAX
from BPM2020.Experiments_Variables import K_EDBN, DIMAURO_PARAMS
from Predictions.DataProcessing import get_data
from Utils.LogFile import LogFile


def train_edbn(data_folder, model_folder, k = None, next_event = True):
    from EDBN.Execute import train
    from Predictions.eDBN_Prediction import learn_duplicated_events, predict_next_event, predict_suffix

    if k is None:
        best_model = {}
        for k in range(1,6):
            train_log = LogFile(data_folder + "train_log.csv", ",", 0, None, None, "case",
                                activity_attr="event", convert=False, k=k)

            train_train_log, train_test_log = train_log.splitTrainTest(80)

            train_train_log.add_end_events()
            train_train_log.convert2int()
            train_train_log.create_k_context()

            train_test_log.values = train_train_log.values
            train_test_log.add_end_events()
            train_test_log.convert2int()
            train_test_log.create_k_context()

            model = train(train_train_log)

            # Train average number of duplicated events
            model.duplicate_events = learn_duplicated_events(train_train_log)

            if next_event:
                acc = predict_next_event(model, train_test_log)
            else:
                acc = predict_suffix(model, train_test_log)
            print("Testing k=", k, " | Validation acc:", acc)
            if "Acc" not in best_model or best_model["Acc"] < acc:
                best_model["Acc"] = acc
                best_model["Model"] = model
                best_model["k"] = k
        print("Best k value:", best_model["k"], " | Validation acc of", best_model["Acc"])
        k = best_model["k"]

    train_log = LogFile(data_folder + "train_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=False, k=k)

    train_log.add_end_events()
    train_log.convert2int()
    train_log.create_k_context()

    model = train(train_log)

    # Train average number of duplicated events
    model.duplicate_events = learn_duplicated_events(train_log)

    with open(os.path.join(model_folder, "model"), "wb") as pickle_file:
        pickle.dump(model, pickle_file)

    with open(os.path.join(model_folder, "k"), "w") as outfile:
        outfile.write(str(k))


def train_camargo(data_folder, model_folder, architecture):
    import RelatedMethods.Camargo.embedding_training as em
    import RelatedMethods.Camargo.model_training as mo

    logfile = LogFile(data_folder + "full_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=True, k=5)
    logfile.create_k_context()
    train_log = LogFile(data_folder + "train_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", values=logfile.values, convert=False, k=5)
    train_log.create_k_context()
    test_log = LogFile(data_folder + "test_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=False, k=5)

    args = {}
    args["file_name"] = "data"
    args["model_type"] = architecture # Choose from 'joint', 'shared', 'concatenated', 'specialized', 'shared_cat'
    args["norm_method"] = "lognorm" # Choose from 'lognorm' or 'max'
    args["n_size"] = 5 # n-gram size
    args['lstm_act'] = None # optimization function see keras doc
    args['l_size'] = 100 # LSTM layer sizes
    args['imp'] = 1 # keras lstm implementation 1 cpu, 2 gpu
    args['dense_act'] = None # optimization function see keras doc
    args['optim'] = 'Nadam' # optimization function see keras doc

    event_emb, role_emb = em.training_model(logfile, model_folder)
    model = mo.training_model(train_log, event_emb, role_emb, args, 200, 10)
    model.save(os.path.join(model_folder, "model.h5"))



def train_lin(data_folder, model_folder):
    from RelatedMethods.Lin.model import create_model

    logfile = LogFile(data_folder + "full_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=False, k=5)
    logfile.add_end_events()
    logfile.convert2int()
    train_log = LogFile(data_folder + "train_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=False, k=5, values=logfile.values)
    train_log.add_end_events()
    train_log.convert2int()
    train_log.create_k_context()

    create_model(train_log, model_folder, 200, 10)


def train_dimauro(data_folder, model_folder, params = None):
    from RelatedMethods.DiMauro.deeppm_act import train
    logfile = LogFile(data_folder + "full_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=False, k=5)
    logfile.add_end_events()
    logfile.convert2int()
    train_log = LogFile(data_folder + "train_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=False, k=5, values=logfile.values)
    train_log.add_end_events()
    train_log.convert2int()
    train_log.create_k_context()

    test_log = os.path.join(data_folder, "test_log.csv")

    train(train_log, test_log, model_folder, params)


def train_tax(data_folder, model_folder):
    from RelatedMethods.Tax.code.train import train

    train_log = os.path.join(data_folder, "train_log.csv")
    test_log = os.path.join(data_folder, "test_log.csv")

    train(train_log, test_log, model_folder)


def main(argv):
    if len(argv) < 2:
        print("Missing arguments, expected: METHOD and DATA")
        return
    else:
        method = argv[0]
        data = argv[1]

    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)

    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    # Check for all input data if already exist. Otherwise, create data
    for train in DATA_DESC:
        dataset_folder = os.path.join(DATA_FOLDER, train["folder"])
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)

            full_logfile, _ = get_data(train["data"], None, 2, False, False, False, False)
            print(full_logfile.data)
            if len(full_logfile.data.columns) == 4:
                full_logfile.data.columns = ["case","event","role", "time"]
            else:
                full_logfile.data.columns = ["case","event","role"]
            full_logfile.trace = "case"
            full_logfile.activity = "event"
            create_csv_file_header(full_logfile.data.to_dict('records'), os.path.join(dataset_folder,'full_log.csv'))

            train_logfile, test_logfile = full_logfile.splitTrainTest(70)
            create_csv_file_header(train_logfile.data.to_dict('records'), os.path.join(dataset_folder,'train_log.csv'))
            create_csv_file_header(test_logfile.data.to_dict('records'), os.path.join(dataset_folder,'test_log.csv'))

        output_folder = os.path.join(OUTPUT_FOLDER, train["folder"])
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
            os.mkdir(os.path.join(output_folder, "models"))

    train = [train for train in DATA_DESC if train["data"] == data][0]
    dataset_folder = os.path.join(DATA_FOLDER, train["folder"])
    model_folder = os.path.join(OUTPUT_FOLDER, str.lower(train["data"]), "models", str.lower(method))

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    edbn_k = K_EDBN

    if method == CAMARGO:
        if len(argv) < 3:
            print("Please indicate the architecture to use: shared_cat or specialized")
            return
        else:
            model_folder = os.path.join(model_folder, argv[2])
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)
    elif method == EDBN:
        if len(argv) >= 3:
            edbn_k = int(argv[2])
        else:
            edbn_k = None
        model_folder = os.path.join(model_folder, str(edbn_k))
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
    ###
    # Register Start time
    ###
    start_time = time.mktime(time.localtime())
    start_time_str = time.strftime("%d-%m-%y %H:%M:%S", time.localtime())
    time_output = open(os.path.join(model_folder, "timings_train.log"), 'a')
    time_output.write("Starting time: %s\n" % start_time_str)

    ###
    # Execute chosen method
    ###
    print("EXPERIMENT TRAINING MODEL:", argv)
    if method == EDBN:
        if len(argv) == 4:
            train_edbn(dataset_folder, model_folder, edbn_k, argv[3] == "next")
        else:
            train_edbn(dataset_folder, model_folder, edbn_k)
    elif method == CAMARGO:
        if len(argv) < 3:
            print("Please indicate the architecture to use: shared_cat or specialized")
            return
        architecture = str.lower(argv[2])
        train_camargo(dataset_folder, model_folder, architecture)
    elif method == LIN:
        train_lin(dataset_folder, model_folder)
    elif method == DIMAURO:
        if len(argv) == 3:
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[2]
        train_dimauro(dataset_folder, model_folder, DIMAURO_PARAMS.get(data, None))
    elif method == TAX:
        train_tax(dataset_folder, model_folder)

    ###
    # Register End time
    ###
    current_time = time.mktime(time.localtime())
    current_time_str = time.strftime("%d-%m-%y %H:%M:%S", time.localtime())
    time_output.write("End time: %s\n" % current_time_str)
    time_output.write("Duration: %fs\n\n" % (current_time - start_time))


if __name__ == "__main__":
    main(sys.argv[1:])

