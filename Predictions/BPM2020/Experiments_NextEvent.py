"""
    Basic file to run the next event prediction experiments
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

from tensorflow.python.keras.models import load_model

from Utils.LogFile import LogFile

OUTPUT_FOLDER = "Output/"


def test_edbn(dataset_folder, model_folder, k):
    from eDBN_Prediction import predict_next_event

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
    acc = sum([1 if a[0] == a[1] else 0 for a in acc]) / len(acc)
    with open(os.path.join(model_folder, "results_next_event.log"), "a") as fout:
        fout.write("Accuracy: (%s) %s\n" % (time.strftime("%d-%m-%y %H:%M:%S", time.localtime()), acc))


def test_camargo(dataset_folder, model_folder, architecture):
    from Methods.Camargo.predict_next import predict_next

    model_file = sorted([model_file for model_file in os.listdir(model_folder) if model_file.endswith(".h5")])[-1]
    model = load_model(os.path.join(model_folder, model_file))

    logfile = LogFile(dataset_folder + "full_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=True, k=5)
    test_log = LogFile(dataset_folder + "test_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=True, k=5, values=logfile.values)
    test_log.create_k_context()

    results = predict_next(test_log, model)
    acc = sum([1 if a[0] == a[1] else 0 for a in results]) / len(results)
    with open(os.path.join(model_folder, "results_next_event.log"), "a") as fout:
        fout.write("Accuracy: (%s) %s\n" % (time.strftime("%d-%m-%y %H:%M:%S", time.localtime()), acc))


def test_lin(dataset_folder, model_folder):
    from Methods.Lin.model import predict_next, Modulator

    logfile = LogFile(dataset_folder + "full_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=True, k=5)
    test_log = LogFile(dataset_folder + "test_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=True, k=5, values=logfile.values)
    test_log.create_k_context()

    model_file = sorted([model_file for model_file in os.listdir(model_folder) if model_file.endswith(".h5")])[-1]
    model = load_model(os.path.join(model_folder, model_file), custom_objects={"Modulator": Modulator})

    acc = predict_next(test_log, model)
    acc = sum([1 if a[0] == a[1] else 0 for a in acc]) / len(acc)
    with open(os.path.join(model_folder, "results_next_event.log"), "a") as fout:
        fout.write("Accuracy: (%s) %s\n" % (time.strftime("%d-%m-%y %H:%M:%S", time.localtime()), acc))


def test_dimauro(dataset_folder, model_folder):
    from Methods.DiMauro.deeppm_act import evaluate

    model_file = sorted([model_file for model_file in os.listdir(model_folder) if model_file.endswith(".h5")])[-1]

    logfile = LogFile(dataset_folder + "full_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=True, k=5)
    train_log = os.path.join(dataset_folder, "train_log.csv")
    test_log = LogFile(dataset_folder + "test_log.csv", ",", 0, None, None, "case",
                        activity_attr="event", convert=True, k=5, values=logfile.values)
    test_log.create_k_context()

    acc = evaluate(train_log, test_log, os.path.join(model_folder, model_file))
    with open(os.path.join(model_folder, "results_next_event.log"), "a") as fout:
        fout.write("Accuracy: (%s) %s\n" % (time.strftime("%d-%m-%y %H:%M:%S", time.localtime()), acc))

def test_tax(dataset_folder, model_folder):
    from Methods.Tax.code.evaluate_next_activity_and_time import evaluate
    from Methods.Tax.code.calculate_accuracy_on_next_event import calc_accuracy

    train_log = os.path.join(dataset_folder, "train_log.csv")
    test_log = os.path.join(dataset_folder, "test_log.csv")
    model_file = sorted([model_file for model_file in os.listdir(model_folder) if model_file.endswith(".h5")])[-1]

    evaluate(train_log, test_log, model_folder, model_file)
    acc = calc_accuracy(os.path.join(model_folder))
    with open(os.path.join(model_folder, "results_next_event.log"), "a") as fout:
        fout.write("Accuracy: (%s) %s\n" % (time.strftime("%d-%m-%y %H:%M:%S", time.localtime()), acc))

def main(argv):
    from Data import get_data
    from Predictions.setting import Setting
    from Methods import get_method
    from Predictions.metric import ACCURACY

    if len(argv) < 2:
        print("Missing arguments, expected: METHOD and DATA")
        return

    method = argv[0]
    data = argv[1]
    d = get_data(data)

    basic_setting = Setting(2, "test-train", False, True, 70, filter_cases=5)

    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    model_folder = os.path.join(OUTPUT_FOLDER, str.lower(d.name), "models", str.lower(method))

    if method == "DBN":
        if method == "DBN":
            if len(argv) >= 3:
                basic_setting.k = int(argv[2])

    if data == "Helpdesk":
        basic_setting.filter_cases = 3

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
    print("EXPERIMENT NEXT ACTIVITY PREDICTION:", argv)
    d.prepare(basic_setting)

    m = get_method(method)

    if method == "CAMARGO":
        if len(argv) < 3:
            print("Please indicate the architecture to use: shared_cat or specialized")
            return
        architecture = str.lower(argv[2])

        m.def_params["model_type"] = architecture

    if method == "DBN":
        model_file = os.path.join(model_folder, "model")
        with open(model_file, "rb") as pickle_file:
            model = pickle.load(pickle_file)
        model.print_parents()
    else:
        model = load_model(os.path.join(model_folder, "model.h5"))


    results = m.test(model, d.test_orig)
    acc = ACCURACY.calculate(results)
    with open(os.path.join(model_folder, "results_next_event.log"), "a") as fout:
        fout.write("Accuracy: (%s) %s\n" % (time.strftime("%d-%m-%y %H:%M:%S", time.localtime()), acc))

    ###
    # Register End time
    ###
    current_time = time.mktime(time.localtime())
    current_time_str = time.strftime("%d-%m-%y %H:%M:%S", time.localtime())
    time_output.write("End time: %s\n" % current_time_str)
    time_output.write("Duration: %fs\n\n" % (current_time - start_time))


if __name__ == "__main__":
    main(sys.argv[1:])