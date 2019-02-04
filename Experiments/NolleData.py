from binet.utils import get_event_logs
from binet.dataset import Dataset

import pandas as pd

import eDBN.Execute as edbn
from LogFile import LogFile
import Utils.PlotResults as plot

SMALL = ["../Data/Nolle_Data/small-0.3-1", "../Data/Nolle_Data/small-0.3-2", "../Data/Nolle_Data/small-0.3-3", "../Data/Nolle_Data/small-0.3-4"]
MEDIUM = ["../Data/Nolle_Data/medium-0.3-1", "../Data/Nolle_Data/medium-0.3-2", "../Data/Nolle_Data/medium-0.3-3", "../Data/Nolle_Data/medium-0.3-4"]
LARGE = ["../Data/Nolle_Data/large-0.3-1", "../Data/Nolle_Data/large-0.3-2", "../Data/Nolle_Data/large-0.3-3", "../Data/Nolle_Data/large-0.3-4"]
HUGE = ["../Data/Nolle_Data/huge-0.3-1", "../Data/Nolle_Data/huge-0.3-2", "../Data/Nolle_Data/huge-0.3-3", "../Data/Nolle_Data/huge-0.3-4"]
P2P = ["../Data/Nolle_Data/wide-0.3-1", "../Data/Nolle_Data/wide-0.3-2", "../Data/Nolle_Data/wide-0.3-3", "../Data/Nolle_Data/wide-0.3-4"]
WIDE = ["../Data/Nolle_Data/p2p-0.3-1", "../Data/Nolle_Data/p2p-0.3-2", "../Data/Nolle_Data/p2p-0.3-3", "../Data/Nolle_Data/p2p-0.3-4"]

def preprocess():
    for log in get_event_logs():
        print(log.name)
        data = Dataset(log.name)

        data.event_log.to_csv("../Data/Nolle_Data/" + log.name + "_data.csv")
        with open("../Data/Nolle_Data/" + log.name + "_labels.csv", "w") as fout:
            for label in data.text_labels:
                if label == "Normal":
                    fout.write("0\n")
                else:
                    fout.write("1\n")


def split_dataset(data_name, label_name, train_name, test_name, train_size):
    log = pd.read_csv(data_name, header=0).drop(columns=["timestamp"])
    labels = pd.read_csv(label_name, header=None)

    normals = set(labels[labels[0] == 0].index + 1)
    anoms = set(labels[labels[0] == 1].index + 1)

    train = log[log.case_id.isin(normals)][:train_size]
    train.to_csv(train_name, index=False)

    test = log[train_size:]
    test["label"] = test.apply(add_label, anoms=anoms, axis=1)
    test.to_csv(test_name, index=False)

def add_label(row, anoms):
    if getattr(row, "case_id") in anoms:
        return 1
    else:
        return 0

def test(files):
    for file in files:
        test_file(file)

def test_file(file):
    split_dataset(file + "_data.csv", file + "_labels.csv", file + "_train.csv", file + "_test.csv", 10000)
    train_data = LogFile(file + "_train.csv", ",", 0, 1000000, None, "case_id", "event")
    train_data.remove_attributes(["event_position"])
    model = edbn.train(train_data)

    test_data = LogFile(file + "_test.csv", ",", 0, 1000000, None, "case_id", "event", values=train_data.values)
    edbn.test(test_data, file + "_output.csv", model, "label", "0")

    plot.plot_single_roc_curve(file + "_output.csv", file, save_file="../Data/Nolle_Graphs/" + file.split("/")[-1] + "_roc.png")
    plot.plot_single_prec_recall_curve(file + "_output.csv", file, save_file="../Data/Nolle_Graphs/" + file.split("/")[-1] + "_precrec.png")


def precision_at_files(files, at, output):
    for file in files:
        precision_at(file, at, output)

def precision_at(file, at, output):
    with open(file + "_output.csv") as finn:
        i = 0
        true_pos = 0
        total_anoms = 0
        for line in finn:
            if i < at and eval(line.split(",")[2]):
                true_pos += 1
            i += 1
            if i == at:
                break
        print("Prec@" + str(at), "for", file + ":", str(true_pos/at))


if __name__ == "__main__":
    # preprocess()
    test(SMALL)
    test(MEDIUM)
    test(LARGE)
    test(HUGE)
    test(P2P)
    test(WIDE)
    precision_at_files(SMALL, 2000, None)
    precision_at_files(MEDIUM, 2000, None)
    precision_at_files(LARGE, 2000, None)
    precision_at_files(HUGE, 2000, None)
    precision_at_files(P2P, 2000, None)
    precision_at_files(WIDE, 2000, None)