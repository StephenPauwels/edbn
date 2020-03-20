import os
import pickle
import sys
import time

from Experiments_Variables import DATA_DESC, DATA_FOLDER, OUTPUT_FOLDER
from Experiments_Variables import EDBN, CAMARGO, DIMAURO, LIN, TAX
from Experiments_Variables import K_EDBN
from Utils.LogFile import LogFile


def test_edbn(dataset_folder, model_folder, k):
    from eDBN_Prediction import predict_next_event

    print("Test EDBN")
    print(dataset_folder)
    print(model_folder)

    model_file = os.path.join(model_folder, "model")

    with open(model_file, "rb") as pickle_file:
        model = pickle.load(pickle_file)
    model.print_parents()

    if k is None:
        with open(os.path.join(model_folder, "k")) as finn:
            k = int(finn.readline())
            print("K=", k)

    train_log = LogFile(dataset_folder + "train_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=True, k=k)

    test_log = LogFile(dataset_folder + "test_log.csv",",", 0, None, None, "case",
                       activity_attr="event", convert=True, k=k, values=train_log.values)
    test_log.create_k_context()

    acc = predict_next_event(model, test_log)
    with open(os.path.join(model_folder, "results_next_event.log"), "a") as fout:
        fout.write("Accuracy: (%s) %s\n" % (time.strftime("%d-%m-%y %H:%M:%S", time.localtime()), acc))


def test_camargo(dataset_folder, model_folder, architecture):
    from Camargo.predict_next import predict_next

    print("Test Camargo")
    model_file = sorted([model_file for model_file in os.listdir(model_folder) if model_file.endswith(".h5")])[-1]

    predict_next(dataset_folder, model_folder, model_file, False)


def test_lin(dataset_folder, model_folder):
    from Lin.model import predict_next

    print("Test Lin")
    logfile = LogFile(dataset_folder + "full_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=True, k=0)
    test_log = LogFile(dataset_folder + "test_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=True, k=0, values=logfile.values)
    model_file = sorted([model_file for model_file in os.listdir(model_folder) if model_file.endswith(".h5")])[-1]

    acc = predict_next(os.path.join(model_folder, model_file), test_log.data, test_log.trace, test_log.activity)
    with open(os.path.join(model_folder, "results_next_event.log"), "a") as fout:
        fout.write("Accuracy: (%s) %s\n" % (time.strftime("%d-%m-%y %H:%M:%S", time.localtime()), acc))


def test_dimauro(dataset_folder, model_folder):
    from DiMauro.deeppm_act import evaluate

    print("Test DiMauro", dataset_folder)
    model_file = sorted([model_file for model_file in os.listdir(model_folder) if model_file.endswith(".h5")])[-1]

    train_log = os.path.join(dataset_folder, "train_log.csv")
    test_log = os.path.join(dataset_folder, "test_log.csv")

    acc = evaluate(train_log, test_log, os.path.join(model_folder, model_file))
    with open(os.path.join(model_folder, "results_next_event.log"), "a") as fout:
        fout.write("Accuracy: (%s) %s\n" % (time.strftime("%d-%m-%y %H:%M:%S", time.localtime()), acc))

def test_tax(dataset_folder, model_folder):
    from Tax.code.evaluate_next_activity_and_time import evaluate
    from Tax.code.calculate_accuracy_on_next_event import calc_accuracy

    print("Test Tax")
    train_log = os.path.join(dataset_folder, "train_log.csv")
    test_log = os.path.join(dataset_folder, "test_log.csv")
    model_file = sorted([model_file for model_file in os.listdir(model_folder) if model_file.endswith(".h5")])[-1]

    evaluate(train_log, test_log, model_folder, model_file)
    acc = calc_accuracy(os.path.join(model_folder))
    with open(os.path.join(model_folder, "results_next_event.log"), "a") as fout:
        fout.write("Accuracy: (%s) %s\n" % (time.strftime("%d-%m-%y %H:%M:%S", time.localtime()), acc))

def main(argv):
    if len(argv) < 2:
        print("Missing arguments, expected: METHOD and DATA")
        return

    method = argv[0]
    data = argv[1]

    train = [train for train in DATA_DESC if train["data"] == data][0]
    dataset_folder = os.path.join(DATA_FOLDER, train["folder"])
    model_folder = os.path.join(OUTPUT_FOLDER, str.lower(train["data"]), "models", str.lower(method))

    edbn_k = K_EDBN

    if method == CAMARGO:
        if len(argv) < 3:
            print("Please indicate the architecture to use: shared_cat or specialized")
            return
        else:
            model_folder = os.path.join(model_folder, argv[2])
    elif method == EDBN:
        if len(argv) >= 3:
            edbn_k = int(argv[2])
        else:
            edbn_k = None
        model_folder = os.path.join(model_folder, str(edbn_k))

    ###
    # Register Start time
    ###
    start_time = time.mktime(time.localtime())
    start_time_str = time.strftime("%d-%m-%y %H:%M:%S", time.localtime())
    time_output = open(os.path.join(model_folder, "timings_next_event.log"), 'a')
    time_output.write("Starting time: %s\n" % start_time_str)

    ###
    # Execute chosen method
    ###
    if method == EDBN:
        test_edbn(dataset_folder, model_folder, edbn_k)
    elif method == CAMARGO:
        if len(argv) < 3:
            print("Please indicate the architecture to use: shared_cat or specialized")
            return
        architecture = str.lower(argv[2])
        test_camargo(dataset_folder, model_folder, architecture)
    elif method == LIN:
        test_lin(dataset_folder, model_folder)
    elif method == DIMAURO:
        test_dimauro(dataset_folder, model_folder)
    elif method == TAX:
        test_tax(dataset_folder, model_folder)

    ###
    # Register End time
    ###
    current_time = time.mktime(time.localtime())
    current_time_str = time.strftime("%d-%m-%y %H:%M:%S", time.localtime())
    time_output.write("End time: %s\n" % current_time_str)
    time_output.write("Duration: %fs\n\n" % (current_time - start_time))


if __name__ == "__main__":
    main(sys.argv[1:])