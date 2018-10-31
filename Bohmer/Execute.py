import sys
sys.path.append("/home/spauwels/PyCharm/anomaly-detection/")
import matplotlib
matplotlib.use('Agg')
import math

import multiprocessing as mp

import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

import Bohmer.LikelihoodGraph as lg


def train(data_file, header = 0, length = 100000):
    lg.clear_variables()
    data = pd.read_csv(data_file, delimiter=",", nrows=length, header=header, dtype=int)
    graph = lg.basicLikelihoodGraph(data, 0)
    V, D = lg.extendLikelihoodGraph(graph, data, 0)
    return (V,D)

def test(train_file, test_file, output_file, model, delim, length, skip=0):
    log = pd.read_csv(train_file, delimiter=",", nrows=length, header=0, dtype=int)
    data = pd.read_csv(test_file, delimiter=delim, nrows=length, header=0, dtype=int, skiprows=skip)
    data.columns = ["Activity", "Resource", "Weekday", "Case", "Anomaly"]
    test_cases = list(data.groupby("Case"))

    scores = Parallel(n_jobs=mp.cpu_count())(
        delayed(lg.test_trace_parallel_for)(model, log, case, 2) for name, case in test_cases)
    scores.sort(key=lambda l: l[1])
    with open(output_file, "w") as fout:
        for s in scores:
            fout.write(",".join([str(i) for i in s]))
            fout.write("\n")
