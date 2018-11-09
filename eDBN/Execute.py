import math
import pandas as pd

import eDBN.GenerateModel as gm

# TODO : Add use of LogFile


def train(data):
    cbn = gm.generate_model(data)
    cbn.train_data(data)
    return cbn

def train2(data_file, trace_attr, header = 0, length = 100000, ignore = None):
    data = pd.read_csv(data_file, delimiter=",", nrows=length, header=header, dtype=int, skiprows=0)
    cbn = gm.generate_model(data, 1, ignore, trace_attr, True)
    cbn.train(data_file, ",", length)
    return cbn

def test(test_data, output_file, model, label, normal_val):
    anoms = model.get_anomalies_sorted(test_data)
    accum_scores = {}
    accum_length = {}
    anomalies = set()
    normal_val = test_data.string_2_int[label][normal_val]
    for i in range(len(anoms)):
        seqID = getattr(anoms[i][1], model.trace_attr)
        if getattr(anoms[i][1], label) != normal_val:
            anomalies.add(seqID)
        if seqID not in accum_scores:
            accum_scores[seqID] = 1
            accum_length[seqID] = 0
        anom_score = 1
        for score in anoms[i][0]:
            anom_score *= score
        accum_scores[seqID] *= anom_score
        accum_length[seqID] += 1
    scores = []

    for seqs in accum_scores:
        scores.append((seqs, math.pow(accum_scores[seqs], 1 / accum_length[seqs]), seqs in anomalies))
    scores.sort(key=lambda l:l[1])
    with open(output_file, "w") as fout:
        for s in scores:
            fout.write(",".join([str(i) for i in s]))
            fout.write("\n")
