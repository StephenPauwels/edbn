"""
    Author: Stephen Pauwels
"""

import numpy as np

import Methods.EDBN.GenerateModel as gm


def train(data, only_activity=True):
    cbn = gm.generate_model(data, only_activity)
    cbn.train(data)
    return cbn


def update(model, data, only_activity=True):
    cbn = gm.update_model(data, only_activity, model)
    cbn.train(data)
    return cbn


def test(test_data, output_file, model, label, normal_val, train_data=None):

    if train_data:
        training_scores = model.test_data(train_data)
        total_training_scores = [score.get_total_score() for score in training_scores]

        mean_training_score = np.mean(total_training_scores)
        std_training_score = np.std(total_training_scores)
        print("EVALUATION: Mean of Training Scores:", mean_training_score)
        print("EVALUATION: Std of Training Scores:", std_training_score)

    anoms = model.test_data(test_data)
    normal_val = test_data.convert_string2int(label, normal_val)
    labels = test_data.get_labels(label)

    # attribute_scores = {"Resource": 10, "Activity": 10, "Weekday": 10, "duration_0": 1}

    scores = []
    for anom in anoms:
        scores.append((test_data.convert_int2string(test_data.trace, anom.id), anom.get_total_score(), labels[anom.id] != normal_val))


    scores.sort(key=lambda l:l[1])

    with open(output_file, "w") as fout:
        for s in scores:
            fout.write(",".join([str(i) for i in s]))
            fout.write("\n")

