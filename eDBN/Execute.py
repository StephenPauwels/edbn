import eDBN.GenerateModel as gm
import matplotlib.pyplot as plt

from Result import Trace_result

def train(data):
    cbn = gm.generate_model(data)
    cbn.train(data)
    return cbn


def test(test_data, output_file, model, label, normal_val, train_data):
    """
    training_scores = model.test_data(train_data)
    attribute_scores = {}
    attrs = training_scores[0].attributes
    for attr in attrs:
        attribute_scores[attr] = []
    for score in training_scores:
        attr_scores = score.get_attribute_scores()
        for attr in attrs:
            attribute_scores[attr].append(attr_scores[attr])

    for attr in attribute_scores:
        print(attribute_scores[attr])
    """

    anoms = model.test_data(test_data)
    accum_scores = {}
    anomalies = set()
    seq_anom_type = {}
    normal_val = test_data.convert_string2int(label, normal_val)
    labels = test_data.get_labels(label)

    scores = []
    for anom in anoms:
        scores.append((anom.id, anom.get_total_score(), labels[anom.id] != normal_val, anom.get_attribute_scores()))
    scores.sort(key=lambda l:l[1])

    y = []

    with open(output_file, "w") as fout:
        for s in scores:
            fout.write(",".join([str(i) for i in s]))
            fout.write("\n")
            y.append(s[1])

    plt.plot(list(range(len(y))), y)
    plt.show()
