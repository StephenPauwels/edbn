import eDBN.GenerateModel as gm
import matplotlib.pyplot as plt
import numpy as np

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
        attribute_scores[attr].sort()
        print(attribute_scores[attr])
    """

    anoms = model.test_data(test_data)
    normal_val = test_data.convert_string2int(label, normal_val)
    labels = test_data.get_labels(label)

    attribute_scores = {"Resource": 10, "Activity": 10, "Weekday": 10, "duration_0": 1}

    duration_anom = test_data.convert_string2int("anom_types", "['alter_duration']")

    duration_scores = []
    scores = []
    for anom in anoms:
        # scores.append((anom.id, anom.get_calibrated_score(attribute_scores), labels[anom.id] != normal_val, anom.get_attribute_scores()))
        dur_scores = anom.get_attribute_score_per_event("duration_0")
        duration_scores.extend(dur_scores)
        act_scores = anom.get_attribute_score_per_event("Activity")

        attr_scores = anom.get_attribute_scores()
#        model_score = attr_scores["Resource"] + attr_scores["Weekday"] + attr_scores["Activity"]
        scores.append((test_data.convert_int2string(test_data.trace, anom.id), anom.get_calibrated_score(attribute_scores), labels[anom.id] != normal_val, test_data.convert_int2string("anom_types", anom.get_anom_type())))#, attr_scores["duration_0"]))
        #scores.append((test_data.convert_int2string("Case_ID", anom.id), anom.get_calibrated_score(attribute_scores), (labels[anom.id] != normal_val) and (anom.get_anom_type() == duration_anom), test_data.convert_int2string("anom_types", anom.get_anom_type())))
        #scores.append((test_data.convert_int2string(test_data.trace, anom.id), attr_scores["duration_0"], (labels[anom.id] != normal_val), test_data.convert_int2string("anom_types", anom.get_anom_type())))

        #scores.append((test_data.convert_int2string("Case_ID", anom.id), anom.get_calibrated_score(attribute_scores), labels[anom.id] != normal_val, anom.get_attribute_scores(), dur_scores, act_scores))
        #scores.append((test_data.convert_int2string("Case_ID", anom.id), anom.get_attribute_score("duration_0"), labels[anom.id] != normal_val, model_score, attr_scores["duration_0"], anom.get_attribute_scores(), dur_scores))
        #scores.append((anom.id, (model_score + 2*attr_scores["duration_0"]) / 3, labels[anom.id] != normal_val, model_score, attr_scores["duration_0"], anom.get_attribute_scores(), dur_scores))


    scores.sort(key=lambda l:l[1])

    with open(output_file, "w") as fout:
        for s in scores:
            fout.write(",".join([str(i) for i in s]))
            fout.write("\n")

