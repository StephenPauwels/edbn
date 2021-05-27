"""
    File containing the Comparison experiments for the ACR Journal

    Author: Stephen Pauwels
"""
import pandas as pd

import Methods.Bohmer.Execute as bohmer
import Methods.EDBN.Execute as edbn
import Utils.PlotResults as plot
from Utils.LogFile import LogFile
from april.dataset import Dataset
from april.fs import get_event_log_files

USED_FILE = 1

SMALL = ["../Data/Nolle_Data/small-0.3-1", "../Data/Nolle_Data/small-0.3-2", "../Data/Nolle_Data/small-0.3-3", "../Data/Nolle_Data/small-0.3-4"]
MEDIUM = ["../Data/Nolle_Data/medium-0.3-1", "../Data/Nolle_Data/medium-0.3-2", "../Data/Nolle_Data/medium-0.3-3", "../Data/Nolle_Data/medium-0.3-4"]
LARGE = ["../Data/Nolle_Data/large-0.3-1", "../Data/Nolle_Data/large-0.3-2", "../Data/Nolle_Data/large-0.3-3", "../Data/Nolle_Data/large-0.3-4"]
HUGE = ["../Data/Nolle_Data/huge-0.3-1", "../Data/Nolle_Data/huge-0.3-2", "../Data/Nolle_Data/huge-0.3-3", "../Data/Nolle_Data/huge-0.3-4"]
WIDE = ["../Data/Nolle_Data/wide-0.3-1", "../Data/Nolle_Data/wide-0.3-2", "../Data/Nolle_Data/wide-0.3-3", "../Data/Nolle_Data/wide-0.3-4"]
P2P = ["../Data/Nolle_Data/p2p-0.3-1", "../Data/Nolle_Data/p2p-0.3-2", "../Data/Nolle_Data/p2p-0.3-3", "../Data/Nolle_Data/p2p-0.3-4"]
GIGANTIC = ["../Data/Nolle_Data/gigantic-0.3-1", "../Data/Nolle_Data/gigantic-0.3-2", "../Data/Nolle_Data/gigantic-0.3-3", "../Data/Nolle_Data/gigantic-0.3-4"]

METHODS = ["DAE", "BINetv1", "BINetv2", "BINetv3"]
###
# Results obtained by executing the Neural Network algorithms
##
SMALL_NOLLE = [(0.50, 0.68), (0.76, 0.91), (0.75, 0.83), (0.80, 0.79)]
MEDIUM_NOLLE = [(0.44, 0.62), (0.68, 0.67), (0.75, 0.73), (0.77, 0.72)]
LARGE_NOLLE = [(0.51, 0.68), (0.79, 0.92), (0.87, 0.81), (0.71, 0.77)]
HUGE_NOLLE = [(0.48, 0.67), (0.71, 0.75), (0.73, 0.72), (0.67, 0.73)]
WIDE_NOLLE = [(0.57, 0.55), (0.70, 0.78), (0.77, 0.80), ((0.75, 0.76))]
P2P_NOLLE = [(0.45, 0.63), (0.74, 0.80), (0.78, 0.80), (0.75, 0.81)]
GIGANTIC_NOLLE = [(0.45, 0.66), (0.71, 0.85), (0.73, 0.73), (0.67, 0.75)]

def preprocess():
    for log in get_event_log_files():
        print(log.name)
        data = Dataset(log.name)

        data.event_log.save_csv("../Data/Nolle_Data/" + log.name + "_data.csv")
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

    if train_size is None:
        log["label"] = log.apply(add_label, anoms=anoms, axis=1)
        log["user"] = log.apply(add_prefix, attr="user", prefix="r_", axis=1)
        log["day"] = log.apply(add_prefix, attr="day", prefix="wd_", axis=1)
        log.to_csv(train_name, index=False)
        log.to_csv(test_name, index=False)
    else:
        train = log[log.case_id.isin(normals)][:train_size]
        train["label"] = train.apply(add_label, anoms=anoms, axis=1)
        train["user"] = train.apply(add_prefix, attr="user", prefix="r_", axis=1)
        train["day"] = train.apply(add_prefix, attr="day", prefix="wd_", axis=1)
        train.to_csv(train_name, index=False)

        test = log
        test["label"] = test.apply(add_label, anoms=anoms, axis=1)
        test["user"] = test.apply(add_prefix, attr="user", prefix="r_", axis=1)
        test["day"] = test.apply(add_prefix, attr="day", prefix="wd_", axis=1)
        test.to_csv(test_name, index=False)

def add_label(row, anoms):
    if getattr(row, "case_id") in anoms:
        return 1
    else:
        return 0

def add_prefix(row, attr, prefix):
    return prefix + str(getattr(row, attr))

def test_sample(files):
    for file in files:
        test_file(file)

def test_file(file):
    split_dataset(file + "_data.csv", file + "_labels.csv", file + "_train.csv", file + "_test.csv", 10000)
    train_data = LogFile(file + "_train.csv", ",", 0, 1000000, None, "case_id", "name")
    train_data.remove_attributes(["label"])
    model = edbn.train(train_data)

    test_data = LogFile(file + "_test.csv", ",", 0, 1000000, None, "case_id", "name", values=train_data.values)
    edbn.test(test_data, file + "_output_sample.csv", model, "label", "0", train_data)

    plot.plot_single_roc_curve(file + "_output_sample.csv", file, save_file="../Data/Nolle_Graphs/" + file.split("/")[-1] + "_roc.png")
    plot.plot_single_prec_recall_curve(file + "_output_sample.csv", file, save_file="../Data/Nolle_Graphs/" + file.split("/")[-1] + "_precrec.png")

def test_full(files):
    for file in files:
        test_file_full(file)

def test_file_full(file):
    split_dataset(file + "_data.csv", file + "_labels.csv", file + "_train.csv", file + "_test.csv", None)
    train_data = LogFile(file + "_train.csv", ",", 0, 1000000, None, "case_id", "name")
    train_data.remove_attributes(["label"])
    model = edbn.train(train_data)

    test_data = LogFile(file + "_test.csv", ",", 0, 1000000, None, "case_id", "name", values=train_data.values)
    edbn.test(test_data, file + "_output_full.csv", model, "label", "0", train_data)

    plot.plot_single_roc_curve(file + "_output_full.csv", file, save_file="../Data/Nolle_Graphs/" + file.split("/")[-1] + "_roc.png")
    plot.plot_single_prec_recall_curve(file + "_output_full.csv", file, save_file="../Data/Nolle_Graphs/" + file.split("/")[-1] + "_precrec.png")

def test_bohmer(files):
    test_file_bohmer(files[0])

def test_file_bohmer(file):
    split_dataset(file + "_data.csv", file + "_labels.csv", file + "_train.csv", file + "_test.csv", 10000)

    train_data = LogFile(file + "_train.csv", ",", 0, 1000000, None, "case_id", "name", convert=False)
    train_data.remove_attributes(["label"])
    model = bohmer.train(train_data, 3, 4, 1)

    test_data = LogFile(file + "_test.csv", ",", 0, 1000000, None, "case_id", "name", convert=False, values=train_data.values)
    bohmer.test(test_data, file + "_output_bohmer.csv", model, "label", 0)

    plot.plot_single_roc_curve(file + "_output_bohmer.csv", file, save_file="../Data/Nolle_Graphs/" + file.split("/")[-1] + "_roc_bohmer.png")
    plot.plot_single_prec_recall_curve(file + "_output_bohmer.csv", file, save_file="../Data/Nolle_Graphs/" + file.split("/")[-1] + "_precrec_bohmer.png")


def compare(files, nolle_result, nolle_labels):
    i = 0
    for file in files:
        results = []
        results.append(file + "_output_sample.csv")
        results.append(file + "_output_full.csv")
        results.append(file + "_output_bohmer.csv")
        plot.plot_compare_prec_recall_curve(results, ["Sample", "Full", "Bohmer"] + nolle_labels , nolle_result, "Comparison", save_file="../Data/Nolle_Graphs/" + file.split("/")[-1] + "_compare_precrec.png")
        plot.plot_compare_roc_curve(results, ["Sample", "Full", "Bohmer"], "Comparison", save_file="../Data/Nolle_Graphs/" + file.split("/")[-1] + "_compare_roc.png")
        i += 1



if __name__ == "__main__":
    preprocess(True)

    test_sample([SMALL[1]])
    test_sample([MEDIUM[1]])
    test_sample([LARGE[1]])
    test_sample([HUGE[1]])
    test_sample([P2P[1]])
    test_sample([WIDE[1]])
    test_sample([GIGANTIC[1]])

    test_full([SMALL[1]])
    test_full([MEDIUM[1]])
    test_full([LARGE[1]])
    test_full([HUGE[1]])
    test_full([P2P[1]])
    test_full([WIDE[1]])
    test_full([GIGANTIC[1]])

    test_bohmer([SMALL[1]])
    test_bohmer([MEDIUM[1]])
    test_bohmer([LARGE[1]])
    test_bohmer([HUGE[1]])
    test_bohmer([P2P[1]])
    test_bohmer([WIDE[1]])
    test_bohmer([GIGANTIC[1]])

    compare([SMALL[1]], SMALL_NOLLE, METHODS)
    compare([MEDIUM[1]], MEDIUM_NOLLE, METHODS)
    compare([LARGE[1]], LARGE_NOLLE, METHODS)
    compare([HUGE[1]], HUGE_NOLLE, METHODS)
    compare([P2P[1]], P2P_NOLLE, METHODS)
    compare([WIDE[1]], WIDE_NOLLE, METHODS)
    compare([GIGANTIC[1]], GIGANTIC_NOLLE, METHODS)
