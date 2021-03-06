"""
    Basic file to run the suffix prediction experiments
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

from BPM2020.Experiments_Variables import DATA_DESC, DATA_FOLDER, OUTPUT_FOLDER
from BPM2020.Experiments_Variables import EDBN, CAMARGO, DIMAURO, LIN, TAX
from BPM2020.Experiments_Variables import K_EDBN
from Utils.LogFile import LogFile


def test_edbn(dataset_folder, model_folder, k=None):
    from eDBN_Prediction import predict_suffix

    model_file = os.path.join(model_folder, "model")

    with open(model_file, "rb") as pickle_file:
        model = pickle.load(pickle_file)
    model.print_parents()

    if k is None:
        with open(os.path.join(model_folder, "k")) as finn:
            k = int(finn.readline())
            print("K=", k)

    train_log = LogFile(dataset_folder + "train_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=False, k=k)
    train_log.add_end_events()
    train_log.convert2int()

    test_log = LogFile(dataset_folder + "test_log.csv",",", 0, None, None, "case",
                       activity_attr="event", convert=False, k=k, values=train_log.values)
    test_log.add_end_events()
    test_log.convert2int()
    test_log.create_k_context()

    acc = predict_suffix(model, test_log)
    with open(os.path.join(model_folder, "results_suffix.log"), "a") as fout:
        fout.write("Accuracy: (%s) %s\n" % (time.strftime("%d-%m-%y %H:%M:%S", time.localtime()), acc))


def test_camargo(dataset_folder, model_folder, architecture):
    from predict_suffix_full import predict_suffix_full

    model_file = sorted([model_file for model_file in os.listdir(model_folder) if model_file.endswith(".h5")])[-1]

    predict_suffix_full(dataset_folder, model_folder, model_file, True)


def test_lin(dataset_folder, model_folder):
    from RelatedMethods.Lin.model import predict_suffix

    logfile = LogFile(dataset_folder + "full_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=False, k=0)
    logfile.add_end_events()
    logfile.convert2int()

    test_log = LogFile(dataset_folder + "test_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=False, k=0, values=logfile.values)
    test_log.add_end_events()
    test_log.convert2int()

    model_file = sorted([model_file for model_file in os.listdir(model_folder) if model_file.endswith(".h5")])[-1]

    acc = predict_suffix(os.path.join(model_folder, model_file), test_log.data, test_log.convert_string2int(test_log.activity, "end"))
    with open(os.path.join(model_folder, "results_suffix.log"), "a") as fout:
        fout.write("Accuracy: (%s) %s\n" % (time.strftime("%d-%m-%y %H:%M:%S", time.localtime()), acc))


def test_dimauro(dataset_folder, model_folder):
    from RelatedMethods.DiMauro.deeppm_act import predict_suffix

    model_file = sorted([model_file for model_file in os.listdir(model_folder) if model_file.endswith(".h5")])[-1]
    acc = predict_suffix(dataset_folder + "train_log.csv", dataset_folder + "test_log.csv", os.path.join(model_folder, model_file))
    with open(os.path.join(model_folder, "results_suffix.log"), "a") as fout:
        fout.write("Accuracy: (%s) %s\n" % (time.strftime("%d-%m-%y %H:%M:%S", time.localtime()), acc))

def test_tax(dataset_folder, model_folder):
    from RelatedMethods.Tax.code.evaluate_suffix_and_remaining_time import evaluate
    from RelatedMethods.Tax.code.calculate_dl_on_suffix import calc_dl

    train_log = os.path.join(dataset_folder, "train_log.csv")
    test_log = os.path.join(dataset_folder, "test_log.csv")
    model_file = sorted([model_file for model_file in os.listdir(model_folder) if model_file.endswith(".h5")])[-1]

    evaluate(train_log, test_log, model_folder, model_file)
    dam_levenstein = calc_dl(os.path.join(model_folder))
    with open(os.path.join(model_folder, "results_suffix.log"), "a") as fout:
        fout.write("Similarity: (%s) %s\n" % (time.strftime("%d-%m-%y %H:%M:%S", time.localtime()), dam_levenstein))

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
    time_output = open(os.path.join(model_folder, "timings_suffix.log"), 'a')
    time_output.write("Starting time: %s\n" % start_time_str)

    ###
    # Execute chosen method
    ###
    print("EXPERIMENT SUFFIX PREDICTION:", argv)
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
        if len(argv) == 3:
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[2]
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