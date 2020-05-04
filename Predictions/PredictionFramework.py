import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys

import time
from multiprocessing import Process

from Utils.LogFile import LogFile

from RelatedMethods.Tax import adapter as tax
from RelatedMethods.Taymouri import adapter as taymouri
from RelatedMethods.Camargo import adapter as camargo
from RelatedMethods.Lin import adapter as lin
from RelatedMethods.DiMauro import adapter as dimauro
from RelatedMethods.Pasquadibisceglie import adapter as pasquadibisceglie
from Predictions import edbn_adapter as edbn
from Predictions import base_adapter as baseline

DATA = ["Camargo_Helpdesk.csv", "Camargo_BPIC12W.csv", "Camargo_BPIC2012.csv", "BPIC15_1_sorted_new.csv",
        "BPIC15_2_sorted_new.csv", "BPIC15_3_sorted_new.csv", "BPIC15_4_sorted_new.csv", "BPIC15_5_sorted_new.csv"]

def run_experiment(data, prefix_size, add_end_event, split_method, split_cases, train_percentage, filename="results.txt"):
    logfile = LogFile(data, ",", 0, None, None, "case",
                      activity_attr="event", convert=False, k=prefix_size)
    if add_end_event:
        logfile.add_end_events()
    logfile.keep_attributes(["case", "event", "role"])
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

    processes = []
    processes.append(Process(target=execute_tax, args=(train_log, test_log, filename), name="Tax"))
    processes.append(Process(target=execute_taymouri, args=(train_log, test_log, filename), name="Taymouri"))
    processes.append(Process(target=execute_camargo, args=(train_log, test_log, filename), name="Camargo"))
    processes.append(Process(target=execute_lin, args=(train_log, test_log, filename), name="Lin"))
    processes.append(Process(target=execute_dimauro, args=(train_log, test_log, filename), name="Di Mauro"))
    processes.append(Process(target=execute_pasquadibisceglie, args=(train_log, test_log, filename), name="Pasquadibisceglie"))
    processes.append(Process(target=execute_edbn, args=(train_log, test_log, filename), name="EDBN"))
    processes.append(Process(target=execute_baseline, args=(train_log, test_log, filename), name="Baseline"))

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
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    baseline_acc = baseline.test(test_log, baseline.train(train_log, epochs=100, early_stop=10))
    with open(filename, "a") as fout:
        fout.write("Baseline: " + str(baseline_acc))
        fout.write("\n")


def execute_edbn(train_log, test_log, filename):
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    edbn_acc = edbn.test(test_log, edbn.train(train_log))
    with open(filename, "a") as fout:
        fout.write("EDBN: " + str(edbn_acc))
        fout.write("\n")


def execute_dimauro(train_log, test_log, filename):
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    dimauro_acc = dimauro.test(test_log, dimauro.train(train_log, epochs=100, early_stop=10))
    with open(filename, "a") as fout:
        fout.write("Di Mauro: " + str(dimauro_acc))
        fout.write("\n")


def execute_lin(train_log, test_log, filename):
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    lin_acc = lin.test(test_log, lin.train(train_log, epochs=100, early_stop=10))
    with open(filename, "a") as fout:
        fout.write("Lin: " + str(lin_acc))
        fout.write("\n")


def execute_camargo(train_log, test_log, filename):
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    camargo_acc = camargo.test(test_log, camargo.train(train_log, epochs=100, early_stop=10))
    with open(filename, "a") as fout:
        fout.write("Camargo: " + str(camargo_acc))
        fout.write("\n")


def execute_taymouri(train_log, test_log, filename):
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    train_data, test_data = taymouri.create_input(train_log, test_log, 5)
    taymouri_acc = taymouri.test(test_data, taymouri.train(train_data))
    with open(filename, "a") as fout:
        fout.write("Taymouri: " + str(taymouri_acc))
        fout.write("\n")


def execute_tax(train_log, test_log, filename):
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    tax_acc = tax.test(test_log, tax.train(train_log, epochs=100, early_stop=10))
    with open(filename, "a") as fout:
        fout.write("Tax: " + str(tax_acc))
        fout.write("\n")


def execute_pasquadibisceglie(train_log, test_log, filename):
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    pasq_acc = pasquadibisceglie.test(test_log, pasquadibisceglie.train(train_log, epochs=100, early_stop=10))
    with open(filename, "a") as fout:
        fout.write("Pasquadibisceglie: " + str(pasq_acc))
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
                        run_experiment(data, ps, aee, sm, sc, tp, filename="test_prefix_size.txt")


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
                            run_experiment(d, ps, aee, sm, sc, tp, filename="test_split_method.txt")


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
                            run_experiment(d, ps, aee, sm, sc, tp, filename="test_end_event.txt")


def experiment_prefix():
    data = DATA
    prefix_size = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
    add_end_event = [True,False]
    split_method = ["train-test"]
    split_cases = [True]
    train_percentage = [70]

    for d in data:
        for ps in prefix_size:
            for aee in add_end_event:
                for sm in split_method:
                    for sc in split_cases:
                        for tp in train_percentage:
                            run_experiment(d, ps, aee, sm, sc, tp, filename="test_prefix.txt")


if __name__ == "__main__":
    experiment_split_method()
    experiment_end_event()
    experiment_prefix()


