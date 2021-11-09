"""
    Author: Stephen Pauwels
"""
import pandas as pd

from Methods.EDBN.Train import train as edbn_train
import Methods.EDBN.Anomalies as edbn
import Utils.PlotResults as plot
from Utils.LogFile import LogFile


def cardio_exec():
    data = "../Data/cardio_data.csv"
    labels = "../Data/cardio_labels.csv"

    log = pd.read_csv(data, header=None)
    labels = pd.read_csv(labels, header=None)
    log["Label"] = labels[0]

    cols = []
    for c in log.columns:
        cols.append("V" + str(c))
    log.columns = cols
    print(log)

    train = log[:250].drop(columns=["VLabel"])
    test = log[250:]

    train.to_csv("../Data/cardio_train.csv", index=False)
    test.to_csv("../Data/cardio_test.csv", index=False)

    train_data = LogFile("../Data/cardio_train.csv", ",", 0, 100000, time_attr=None, trace_attr=None, convert=False, dtype="float64")
    test_data = LogFile("../Data/cardio_test.csv", ",", 0, 100000, time_attr=None, trace_attr=None, convert=False, dtype="float64")

    """ # Discretization
    attr_bins = {}
    bins = 10
    labels = [str(i) for i in range(1, bins + 1)]
    for attr in train_data.numericalAttributes:
        train_data.data[attr], attr_bins[attr] = pd.cut(train_data.data[attr], bins, retbins=True, labels=labels)

    for attr in train_data.numericalAttributes:
        train_data.categoricalAttributes.add(attr)

    for attr in test_data.numericalAttributes:
        if attr != "VLabel":
            labels = [str(i) for i in range(1, len(attr_bins[attr]))]
            test_data.data[attr] = pd.cut(test_data.data[attr], attr_bins[attr], retbins=False, labels=labels)

    for attr in test_data.numericalAttributes:
        test_data.categoricalAttributes.add(attr)

    train_data.numericalAttributes = set()
    test_data.numericalAttributes = set()
    """

    model = edbn_train(train_data)
    edbn.test(test_data, "../Data/cardio_output_num.csv", model, label="VLabel", normal_val=0)


def mammo_exec():
    data = "../Data/mammo_data.csv"
    labels = "../Data/mammo_labels.csv"

    log = pd.read_csv(data, header=None)
    labels = pd.read_csv(labels, header=None)
    log["Label"] = labels[0]

    cols = []
    for c in log.columns:
        cols.append("V" + str(c))
    log.columns = cols
    print(log)

    train = log[:5000]
    test = log[5000:]

    train = train[train.VLabel == 0].drop(columns=["VLabel"])

    train.to_csv("../Data/mammo_train.csv", index=False)
    test.to_csv("../Data/mammo_test.csv", index=False)

    train_data = LogFile("../Data/mammo_train.csv", ",", 0, 100000, time_attr=None, trace_attr=None, convert=False, dtype="float64")
    test_data = LogFile("../Data/mammo_test.csv", ",", 0, 100000, time_attr=None, trace_attr=None, convert=False, dtype="float64")

    """ # Discretization
    attr_bins = {}
    bins = 10
    labels = [str(i) for i in range(1, bins + 1)]
    for attr in train_data.numericalAttributes:
        train_data.data[attr], attr_bins[attr] = pd.cut(train_data.data[attr], bins, retbins=True, labels=labels)

    for attr in train_data.numericalAttributes:
        train_data.categoricalAttributes.add(attr)

    for attr in test_data.numericalAttributes:
        if attr != "VLabel":
            labels = [str(i) for i in range(1, len(attr_bins[attr]))]
            test_data.data[attr] = pd.cut(test_data.data[attr], attr_bins[attr], retbins=False, labels=labels)

    for attr in test_data.numericalAttributes:
        test_data.categoricalAttributes.add(attr)

    train_data.numericalAttributes = set()
    test_data.numericalAttributes = set()
    """

    model = edbn_train(train_data)
    edbn.test(test_data, "../Data/mammo_output_num.csv", model, label="VLabel", normal_val=0)


def breast_exec():
    data = "../Data/breast_data.csv"
    labels = "../Data/breast_labels.csv"

    log = pd.read_csv(data, header=None)
    labels = pd.read_csv(labels, header=None)
    log["Label"] = labels[0]

    cols = []
    for c in log.columns:
        cols.append("V" + str(c))
    log.columns = cols
    print(log)

    train = log[:100]
    test = log[100:]

    train = train[train.VLabel == 0].drop(columns=["VLabel"])

    train.to_csv("../Data/breast_train.csv", index=False)
    test.to_csv("../Data/breast_test.csv", index=False)

    train_data = LogFile("../Data/breast_train.csv", ",", 0, 100000, time_attr=None, trace_attr=None, convert=False, dtype="float64")
    test_data = LogFile("../Data/breast_test.csv", ",", 0, 100000, time_attr=None, trace_attr=None, convert=False, dtype="float64")

    """ # Discretization
    attr_bins = {}
    bins = 10
    labels = [str(i) for i in range(1, bins + 1)]
    for attr in train_data.numericalAttributes:
        train_data.data[attr], attr_bins[attr] = pd.cut(train_data.data[attr], bins, retbins=True, labels=labels)

    for attr in train_data.numericalAttributes:
        train_data.categoricalAttributes.add(attr)
    train_data.numericalAttributes = set()

    for attr in test_data.numericalAttributes:
        if attr != "VLabel":
            labels = [str(i) for i in range(1, len(attr_bins[attr]))]
            test_data.data[attr] = pd.cut(test_data.data[attr], attr_bins[attr], retbins=False, labels=labels)

    for attr in test_data.numericalAttributes:
        test_data.categoricalAttributes.add(attr)
    test_data.numericalAttributes = set()
    """

    model = edbn_train(train_data)
    edbn.test(test_data, "../Data/breast_output_num.csv", model, label="VLabel", normal_val=0)

if __name__ == "__main__":
    cardio_exec()
    mammo_exec()
    breast_exec()

    # !! Make sure all output files where generated
    files_out = ["../Data/breast_output.csv", "../Data/breast_output_num.csv", "../Data/cardio_output.csv",
                 "../Data/cardio_output_num.csv","../Data/mammo_output.csv", "../Data/mammo_output_num.csv"]
    labels = ["Breast (Disc)", "Breast", "Cardio (Disc)", "Cardio", "Mammo (Disc)", "Mammo"]
    plot.plot_compare_roc_curve(files_out, labels, title="Continuous Datasets", save_file="../Data/Continuous_ROC_Disc.png")
    plot.plot_compare_prec_recall_curve(files_out, labels, title="Continuous Datasets", save_file="../Data/Continuous_PR_Disc.png")
