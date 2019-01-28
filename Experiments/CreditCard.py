import pandas as pd
import BayesianNet as bn
import numpy as np
import Utils.PlotResults as plot
import matplotlib.pyplot as plt

from extended_Dynamic_Bayesian_Network import extendedDynamicBayesianNetwork
import Uncertainty_Coefficient as uc


def convert2ints(data, values):
    """
    Convert csv file with string values to csv file with integer values.
    (File/string operations more efficient than pandas operations)

    :param file_out: filename for newly created file
    :return: number of lines converted
    """
    data = data.apply(lambda x: convert_column2ints(x, values))
    return data


def convert_column2ints(x, values):
    def test(a, b):
        # Return all elements from a that are not in b, make use of the fact that both a and b are unique and sorted
        a_ix = 0
        b_ix = 0
        new_uniques = []
        while a_ix < len(a) and b_ix < len(b):
            if a[a_ix] < b[b_ix]:
                new_uniques.append(a[a_ix])
                a_ix += 1
            elif a[a_ix] > b[b_ix]:
                b_ix += 1
            else:
                a_ix += 1
                b_ix += 1
        if a_ix < len(a):
            new_uniques.extend(a[a_ix:])
        return new_uniques

    print("PREPROCESSING: Converting", x.name)
    if x.name not in values:
        x = x.astype("str")
        values[x.name], y = np.unique(x, return_inverse=True)
        return y + 1
    else:
        x = x.astype("str")
        values[x.name] = np.append(values[x.name], test(np.unique(x), values[x.name]))

        print("PREPROCESSING: Substituting values with ints")
        xsorted = np.argsort(values[x.name])
        ypos = np.searchsorted(values[x.name][xsorted], x)
        indices = xsorted[ypos]

    return indices + 1

if __name__ == "__main__":
    log = pd.read_csv("../Data/creditcard.csv", nrows=100000, dtype='float64').drop(columns=["Time"])
    for col in log:
        if col != "Class":
            log[col] = pd.cut(log[col], bins=5) # Discretize all attributes in n bins

    for col in log:
        print(col)
        print(log[col].value_counts())

    values = {}
    log = convert2ints(log, values)

    print(log["Class"].value_counts())

    train = log[:10000]
    test = log[10000:]

    train = train[train.Class == 1] # Only keep non-anomalies
    train = train.drop(columns=["Class"]) # Drop Class label

    edbn = extendedDynamicBayesianNetwork(len(train.columns), 0, None)

    # Calculate New_value
    for col in train.columns:
        new_vals = uc.calculate_new_values_rate(train[col])
        edbn.add_variable(col, new_vals, None)

    mappings = uc.calculate_mappings(train, train.columns, 0, 0.99)
    print(mappings)

    bay_net = bn.BayesianNetwork(train)
    net = bay_net.hill_climbing_pybn(train.columns, metric="AIC", whitelist=[])

    relations = []
    for edge in net.edges():
        relations.append((edge[0], edge[1]))

    for relation in relations:
    #    if relation not in mappings:
        edbn.add_parent(relation[0], relation[1])
        print(relation[0], "->", relation[1])

    model = edbn.train(train, single=True)

    ranking = edbn.test(test)
    ranking.sort(key=lambda l: l[0].get_total_score())
    scores = []
    y = []
    for r in ranking:
        scores.append((getattr(r[1], "Index"), r[0].get_total_score(), getattr(r[1], "Class") != 1))
        y.append(r[0].get_total_score())
    print(len(scores))

    with open("../output.csv", "w") as fout:
        for s in scores:
            fout.write(",".join([str(i) for i in s]))
            fout.write("\n")

    plot.plot_single_roc_curve("../output.csv")
    plot.plot_single_prec_recall_curve("../output.csv")

    plt.plot(list(range(len(y))), y)
    plt.show()

