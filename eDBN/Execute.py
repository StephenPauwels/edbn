import eDBN.GenerateModel as gm

def train(data):
    cbn = gm.generate_model(data)
    cbn.train(data)
    return cbn


def test(test_data, output_file, model, label, normal_val):
    anoms = model.test_data(test_data)
    accum_scores = {}
    accum_length = {}
    anomalies = set()
    seq_anom_type = {}
    normal_val = test_data.convert_string2int(label, normal_val)
    for i in range(len(anoms)):
        seqID = getattr(anoms[i][1], model.trace_attr)

        #seq_anom_type[seqID] = test_data.convert_int2string("anom_types", int(getattr(anoms[i][1], "anom_types")))
        if getattr(anoms[i][1], label) != normal_val:
            anomalies.add(seqID)

        if seqID not in accum_scores:
            accum_scores[seqID] = 0
            accum_length[seqID] = 0

        accum_scores[seqID] += anoms[i][0].get_total_score()
        accum_length[seqID] += 1
    scores = []

    for seqs in accum_scores:
        scores.append((seqs, accum_scores[seqs] / accum_length[seqs], seqs in anomalies))#, seq_anom_type[seqs]))
    scores.sort(key=lambda l:l[1])

    with open(output_file, "w") as fout:
        for s in scores:
            fout.write(",".join([str(i) for i in s]))
            fout.write("\n")
