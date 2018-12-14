import sys
#sys.path.append("/home/spauwels/PyCharm/anomaly-detection/")
import matplotlib
#matplotlib.use('Agg')
import math

import multiprocessing as mp

import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

import Bohmer.LikelihoodGraph as lg


def train(data):

    model = lg.LikelihoodModel(data)

    model.basicLikelihoodGraph()
    model.extendLikelihoodGraph()
    return model

def test(train, test, output_file, model):

    test_cases = list(test.data.groupby("Case"))

    #scores = Parallel(n_jobs=mp.cpu_count())(
    #    delayed(lg.test_trace_parallel_for)(model, log, case, 2, lg.global_dict_to_value) for name, case in test_cases)
    scores = []
    for name, case in test_cases:
        scores.append((name, model.test_trace(case), case.iloc[0]["Anomaly"] == "1"))

    scores.sort(key=lambda l: l[1])
    with open(output_file, "w") as fout:
        for s in scores:
            fout.write(",".join([str(i) for i in s]))
            fout.write("\n")
