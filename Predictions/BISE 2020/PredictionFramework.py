import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys

import time
from multiprocessing import Process

from Utils.LogFile import LogFile

# DATA = ["Helpdesk.csv", "BPIC12W.csv", "BPIC12.csv", "BPIC15_1_sorted_new.csv",
#         "BPIC15_2_sorted_new.csv", "BPIC15_3_sorted_new.csv", "BPIC15_4_sorted_new.csv", "BPIC15_5_sorted_new.csv"]
DATA = ["BPIC15_1_sorted_new.csv",
        "BPIC15_2_sorted_new.csv", "BPIC15_3_sorted_new.csv", "BPIC15_4_sorted_new.csv", "BPIC15_5_sorted_new.csv"]
DATA_FOLDER = "../../Data/"

def run_experiment(data, prefix_size, add_end_event, split_method, split_cases, train_percentage, filename="results.txt"):
    data = DATA_FOLDER + data
    logfile = LogFile(data, ",", 0, None, "completeTime", "case",
                      activity_attr="event", convert=False, k=prefix_size)

    if prefix_size is None:
        prefix_size = max(logfile.data.groupby(logfile.trace).size())
        if prefix_size > 40:
            prefix_size = 40
    logfile.k = prefix_size

    if add_end_event:
        logfile.add_end_events()
    logfile.keep_attributes(["case", "event", "role", "completeTime"])
    # logfile.keep_attributes(["case", "event", "role"])
    logfile.convert2int()
    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(train_percentage, case=split_cases, method=split_method)

    with open(filename, "a") as fout:
        fout.write("Data: " + data)
        fout.write("\nPrefix Size: " + str(prefix_size))
        fout.write("\nEnd event: " + str(add_end_event))
        fout.write("\nSplit method: " + split_method)
        fout.write("\nSplit cases: " + str(split_cases))
        fout.write("\nTrain percentage: " + str(train_percentage))
        fout.write("\nDate: " + time.strftime("%d.%m.%y-%H.%M", time.localtime()))
        fout.write("\n------------------------------------\n")

    data_folder = filename.replace(".txt", "") + "_" + str(DATA.index(data.replace(DATA_FOLDER, "")))
    processes = []
    # processes.append(Process(target=execute_tax, args=(train_log, test_log, filename), name="Tax"))
    # processes.append(Process(target=execute_taymouri, args=(train_log, test_log, filename), name="Taymouri"))
    # processes.append(Process(target=execute_camargo, args=(train_log, test_log, filename), name="Camargo"))
    # processes.append(Process(target=execute_lin, args=(train_log, test_log, filename), name="Lin"))
    # processes.append(Process(target=execute_dimauro, args=(train_log, test_log, filename), name="Di Mauro"))
    # processes.append(Process(target=execute_pasquadibisceglie, args=(train_log, test_log, filename), name="Pasquadibisceglie"))
    processes.append(Process(target=execute_pasquadibisceglie2020, args=(train_log, test_log, filename, data_folder),
                             name="Pasquadibisceglie (2020)"))
    # processes.append(Process(target=execute_edbn, args=(train_log, test_log, filename), name="EDBN"))
    # processes.append(Process(target=execute_edbn_update, args=(train_log, test_log, filename), name="EDBN_Update"))
    # processes.append(Process(target=execute_baseline, args=(train_log, test_log, filename), name="Baseline"))
    # processes.append(Process(target=execute_new_method, args=(train_log, test_log, filename), name="New Method"))

    print("Starting Processes")
    for p in processes:
        p.start()
        print(p.name, "started")

    print("All processes running")

    for p in processes:
        p.join()
        print(p.name, "stopped")

    with open(filename, "a") as fout:
        fout.write("====================================\n\n")

    print("All processes stopped")


def execute_baseline(train_log, test_log, filename):
    from Predictions import base_adapter as baseline

    sys.stdout = open("../log/baseline.out", "a")
    sys.stderr = open("../log/baseline.error", "a")
    baseline_acc = baseline.test(test_log, baseline.train(train_log, epochs=100, early_stop=10))
    with open(filename, "a") as fout:
        fout.write("Baseline: " + str(baseline_acc))
        fout.write("\n")

def execute_new_method(train_log, test_log, filename):
    from Predictions import new_method_adapter as new_method

    sys.stdout = open("../log/new_method.out", "a")
    sys.stderr = open("../log/new_method.error", "a")
    baseline_acc = new_method.test(test_log, new_method.train(train_log, epochs=100, early_stop=10))
    with open(filename, "a") as fout:
        fout.write("New: " + str(baseline_acc))
        fout.write("\n")


def execute_edbn(train_log, test_log, filename):
    from Predictions import edbn_adapter as edbn

    # sys.stdout = open("log/edbn.out", "a")
    # sys.stderr = open("log/edbn.error", "a")
    train_log.keep_attributes(["case", "event", "role"])
    test_log.keep_attributes(["case", "event", "role"])
    edbn_acc = edbn.test(test_log, edbn.train(train_log))
    with open(filename, "a") as fout:
        fout.write("EDBN: " + str(edbn_acc))
        fout.write("\n")

def execute_edbn_update(train_log, test_log, filename):
    from Predictions import edbn_adapter as edbn
    train_log.keep_attributes(["case", "event", "role"])
    test_log.keep_attributes(["case", "event", "role"])
    edbn_acc = edbn.test_and_update(test_log, edbn.train(train_log))
    with open(filename, "a") as fout:
        fout.write("EDBN_update: " + str(edbn_acc))
        fout.write("\n")


def execute_dimauro(train_log, test_log, filename):
    from RelatedMethods.DiMauro import adapter as dimauro

    sys.stdout = open("../log/dimauro.out", "a")
    sys.stderr = open("../log/dimauro.error", "a")
    dimauro_acc = dimauro.test(test_log, dimauro.train(train_log, epochs=100, early_stop=10))
    with open(filename, "a") as fout:
        fout.write("Di Mauro: " + str(dimauro_acc))
        fout.write("\n")


def execute_lin(train_log, test_log, filename):
    from RelatedMethods.Lin import adapter as lin

    sys.stdout = open("../log/lin.out", "a")
    sys.stderr = open("../log/lin.error", "a")
    lin_acc = lin.test(test_log, lin.train(train_log, epochs=100, early_stop=10))
    with open(filename, "a") as fout:
        fout.write("Lin: " + str(lin_acc))
        fout.write("\n")


def execute_camargo(train_log, test_log, filename):
    from RelatedMethods.Camargo import adapter as camargo

    sys.stdout = open("../log/camargo.out", "a")
    sys.stderr = open("../log/camargo.error", "a")
    camargo_acc = camargo.test(test_log, camargo.train(train_log, epochs=100, early_stop=10))
    with open(filename, "a") as fout:
        fout.write("Camargo: " + str(camargo_acc))
        fout.write("\n")


def execute_taymouri(train_log, test_log, filename):
    from RelatedMethods.Taymouri import adapter as taymouri

    sys.stdout = open("../log/taymouri.out", "a")
    sys.stderr = open("../log/taymouri.error", "a")
    train_data, test_data = taymouri.create_input(train_log, test_log, 5)
    taymouri_acc = taymouri.test(test_data, taymouri.train(train_data))
    with open(filename, "a") as fout:
        fout.write("Taymouri: " + str(taymouri_acc))
        fout.write("\n")


def execute_tax(train_log, test_log, filename):
    from RelatedMethods.Tax import adapter as tax

    sys.stdout = open("../log/tax.out", "a")
    sys.stderr = open("../log/tax.error", "a")
    tax_acc = tax.test(test_log, tax.train(train_log, epochs=100, early_stop=10))
    with open(filename, "a") as fout:
        fout.write("Tax: " + str(tax_acc))
        fout.write("\n")


def execute_pasquadibisceglie(train_log, test_log, filename):
    from RelatedMethods.Pasquadibisceglie import adapter as pasquadibisceglie

    sys.stdout = open("../log/pasquadibisceglie.out", "a")
    sys.stderr = open("../log/pasquadibisceglie.error", "a")
    pasq_acc = pasquadibisceglie.test(test_log, pasquadibisceglie.train(train_log, epochs=100, early_stop=10))
    with open(filename, "a") as fout:
        fout.write("Pasquadibisceglie: " + str(pasq_acc))
        fout.write("\n")


def execute_pasquadibisceglie2020(train_log, test_log, filename, data_folder=None):
    from RelatedMethods.Premiere import adapter as pasquadibisceglie

    # sys.stdout = open("../log/pasquadibisceglie2020.out", "a")
    # sys.stderr = open("../log/pasquadibisceglie2020.error", "a")
    pasq_acc = pasquadibisceglie.test(test_log, pasquadibisceglie.train(train_log, epochs=100, early_stop=10,
                                                                        folder=data_folder + "_train"), data_folder + "_test")
    with open(filename, "a") as fout:
        fout.write("Pasquadibisceglie (2020): " + str(pasq_acc))
        fout.write("\n")

def experiments_helpdesk():
    data = "../Data/BPIC15_5_sorted_new.csv"
    prefix_size = [1, 5, 10, 15, 20, 25, 30, 35]
    add_end_event = [False]
    split_method = ["train-test"]
    split_cases = [True]
    train_percentage = [70]

    for ps in prefix_size:
        for aee in add_end_event:
            for sm in split_method:
                for sc in split_cases:
                    for tp in train_percentage:
                        run_experiment(data, ps, aee, sm, sc, tp, filename="../test_prefix_size.txt")


def experiment_split_method():
    data = DATA
    prefix_size = [10]
    add_end_event = [False]
    split_method = ["random", "train-test", "test-train"]
    split_cases = [True]
    train_percentage = [70]

    for d in data:
        for ps in prefix_size:
            for aee in add_end_event:
                for sm in split_method:
                    for sc in split_cases:
                        for tp in train_percentage:
                            run_experiment(d, ps, aee, sm, sc, tp, filename="../test_split_method.txt")


def experiment_end_event():
    data = DATA
    prefix_size = [10]
    add_end_event = [True, False]
    split_method = ["train-test"]
    split_cases = [True]
    train_percentage = [70]

    for d in data:
        for ps in prefix_size:
            for aee in add_end_event:
                for sm in split_method:
                    for sc in split_cases:
                        for tp in train_percentage:
                            run_experiment(d, ps, aee, sm, sc, tp, filename="../test_end_event2.txt")


def experiment_prefix():
    data = DATA
    prefix_size = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
    add_end_event = [False]
    split_method = ["train-test"]
    split_cases = [True]
    train_percentage = [70]

    for d in data:
        for ps in prefix_size:
            for aee in add_end_event:
                for sm in split_method:
                    for sc in split_cases:
                        for tp in train_percentage:
                            run_experiment(d, ps, aee, sm, sc, tp, filename="../../Predictions_Results/Prefix Size/test_prefix_dimauro.txt")

def experiment_bpm2020():
    data = ["BPIC15_1_sorted_new.csv", "BPIC15_2_sorted_new.csv", "BPIC15_3_sorted_new.csv", "BPIC15_4_sorted_new.csv", "BPIC15_5_sorted_new.csv"]
    for d in data:
        run_experiment(d, 5, False, "test-train", True, 70, filename="../test_bpm_dimauro.txt")


def all_experiments():
    data = ["Helpdesk.csv", "BPIC12W.csv", "BPIC12.csv", "BPIC15_1_sorted_new.csv",
            "BPIC15_3_sorted_new.csv", "BPIC15_5_sorted_new.csv"]
    # prefix_size = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
    prefix_size = [40]
    add_end_event = [True, False]
    split_method = ["train-test", "test-train", "random"]
    split_cases = [True, False]
    train_percentage = [70, 80]

    for d in data:
        for ps in prefix_size:
            for aee in add_end_event:
                for sm in split_method:
                    for sc in split_cases:
                        for tp in train_percentage:
                            run_experiment(d, ps, aee, sm, sc, tp, filename="Baseline/test_baseline_3.txt")


def paper_experiments():
    configs = []
    for d in DATA:
        configs.append({"data": d, "prefix_size": None, "add_end_event": True, "split_method": "train-test",
                        "split_cases": False, "train_percentage": 66, "filename": "paper_tax_extra.txt"})

        configs.append({"data": d, "prefix_size": 5, "add_end_event": True, "split_method": "test-train",
                        "split_cases": False, "train_percentage": 70, "filename": "paper_camargo_extra.txt"})

        configs.append({"data": d, "prefix_size": None, "add_end_event": True, "split_method": "train-test",
                        "split_cases": False, "train_percentage": 70, "filename": "paper_lin_extra.txt"})

        configs.append({"data": d, "prefix_size": None, "add_end_event": True, "split_method": "random",
                        "split_cases": False, "train_percentage": 80, "filename": "paper_dimauro_extra.txt"})

        configs.append({"data": d, "prefix_size": None, "add_end_event": True, "split_method": "train-test",
                        "split_cases": False, "train_percentage": 66, "filename": "paper_pasquadibisceglie_extra.txt"})

        configs.append({"data": d, "prefix_size": 5, "add_end_event": True, "split_method": "train-test",
                        "split_cases": True, "train_percentage": 80, "filename": "paper_taymouri_extra.txt"})

        configs.append({"data": d, "prefix_size": 10, "add_end_event": False, "split_method": "train-test",
                        "split_cases": True, "train_percentage": 70, "filename": "paper_baseline_extra.txt"})

    for config in configs:
        run_experiment(**config)

if __name__ == "__main__":
    # recreate experiments used in BPM Forum 2020 paper
    # experiment_bpm2020()
    #
    # # perform separate experiments on influence of preprocessing choices
    # experiment_split_method()
    # experiment_end_event()
    # experiment_prefix()
    #
    # # perform all experiments with all possible preprocessing choices
    # all_experiments()

    # perform ranking experiments, using the settings described in the different papers
    paper_experiments()

