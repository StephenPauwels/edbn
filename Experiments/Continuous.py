import pandas as pd
import hill_climbing_continuous as hcc
from LogFile import LogFile
import eDBN.Execute as edbn
import Utils.PlotResults as plot


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

    model = hcc.learn_continuous_net(train)
    hcc.score_continuous_net(model, test, "VLabel", "../Data/cardio_output.csv", "cardio")

    print(log["VLabel"].value_counts())
    print(test["VLabel"].value_counts())


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


    model = hcc.learn_continuous_net(train)
    hcc.score_continuous_net(model, test, "VLabel", "../Data/mammo_output.csv","mammo")

    print(log["VLabel"].value_counts())
    print(test["VLabel"].value_counts())

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

    model = hcc.learn_continuous_net(train)

    print(log["VLabel"].value_counts())
    print(test["VLabel"].value_counts())

    hcc.score_continuous_net(model, test, "VLabel", "../Data/breast_output.csv", "breast")

def breast_discrete_exec():
    data = "../Data/breast_data.csv"
    labels = "../Data/breast_labels.csv"

    log = pd.read_csv(data, header=None)
    labels = pd.read_csv(labels, header=None)
    log["Label"] = labels[0]

    cols = []
    for c in log.columns:
        cols.append("V" + str(c))
    log.columns = cols
    log['ID'] = log.reset_index().index
    print(log)

    train = log[:100]
    test = log[100:]
    train = train[train.VLabel == 0].drop(columns=["VLabel"])

    train.to_csv("../Data/breast_train.csv", index=False)
    test.to_csv("../Data/breast_test.csv", index=False)

    train_data = LogFile("../Data/breast_train.csv", ",", 0, 500000, None, "ID", activity_attr="Activity")
    train_data.k = 0
    model = edbn.train(train_data)

    test_data = LogFile("../Data/breast_test.csv", ",", 0, 500000, None, "ID", activity_attr="Activity")
    test_data.k = 0
    print(test_data.data)
    edbn.test(test_data, "../Data/breast_discrete_output.csv", model, "VLabel", "0")

    plot.plot_single_roc_curve("../Data/breast_discrete_output.csv", "breast_discrete")
    plot.plot_single_prec_recall_curve("../Data/breast_discrete_output.csv", "breast_discrete")


def letter_exec():
    data = "../Data/letter_data.csv"
    labels = "../Data/letter_labels.csv"

    log = pd.read_csv(data, header=None)
    labels = pd.read_csv(labels, header=None)
    log["Label"] = labels[0]

    cols = []
    for c in log.columns:
        cols.append("V" + str(c))
    log.columns = cols
    print(log)

    train = log[:300]
    test = log[300:]

    train = train[train.VLabel == 0].drop(columns=["VLabel"])

    model = hcc.learn_continuous_net(train)

    print(log["VLabel"].value_counts())
    print(test["VLabel"].value_counts())

    hcc.score_continuous_net(model, test, "VLabel", "../Data/letter_output.csv", "letter")

def letter_discrete_exec():
    data = "../Data/letter_data.csv"
    labels = "../Data/letter_labels.csv"

    log = pd.read_csv(data, header=None)
    labels = pd.read_csv(labels, header=None)
    log["Label"] = labels[0]

    cols = []
    for c in log.columns:
        cols.append("V" + str(c))
    log.columns = cols
    log['ID'] = log.reset_index().index
    print(log)

    train = log[:300]
    test = log[300:]
    train = train[train.VLabel == 0].drop(columns=["VLabel"])

    train.to_csv("../Data/letter_train.csv", index=False)
    test.to_csv("../Data/letter_test.csv", index=False)

    train_data = LogFile("../Data/letter_train.csv", ",", 0, 500000, None, "ID", activity_attr="Activity")
    train_data.k = 0
    model = edbn.train(train_data)

    test_data = LogFile("../Data/letter_test.csv", ",", 0, 500000, None, "ID", activity_attr="Activity")
    test_data.k = 0
    print(test_data.data)
    edbn.test(test_data, "../Data/letter_discrete_output.csv", model, "VLabel", "0")

    plot.plot_single_roc_curve("../Data/letter_discrete_output.csv", 'letter_discrete')
    plot.plot_single_prec_recall_curve("../Data/letter_discrete_output.csv", 'letter_discrete')

if __name__ == "__main__":
    cardio_exec()
    #mammo_exec()
    breast_exec()
    #breast_discrete_exec()
    #letter_exec()
    #letter_discrete_exec()