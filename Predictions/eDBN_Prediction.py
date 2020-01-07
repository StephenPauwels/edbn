import copy

import EDBN.Execute as edbn

import Preprocessing as data
import re

import numpy as np
import jellyfish as jf
import multiprocessing as mp
import functools
import matplotlib.pyplot as plt
import random
import math


def get_probabilities(variable, val_tuple, parents):
    if val_tuple in variable.cpt:
        return variable.cpt[val_tuple], False
    else:
        # Check if single values occur in dataset
        value_probs = []
        for i in range(len(val_tuple)):
            if val_tuple[i] not in parents[i].values:
                value_probs.append(None)
            else:
                value_probs.append(parents[i].values[val_tuple[i]])

        if None in value_probs:
            # TODO iterate over all possible values of None valued attributes
            possible_values = [[]]
            for idx in range(len(parents)):
                if value_probs[idx] is None:
                    new_possible_values = []
                    for poss_val in possible_values:
                        for val in parents[idx].values.keys():
                                new_possible_values.append(poss_val + [val])
                    possible_values = new_possible_values
                else:
                    for poss_val in possible_values:
                        poss_val.append(val_tuple[idx])

            prediction_options = {}

            for value in possible_values:
                value_prob = 1
                for idx in range(len(parents)):
                    if value_probs[idx] is None:
                        value_prob *= parents[idx].values[value[idx]]
                parent_tuple_val = tuple(value)
                if parent_tuple_val in variable.cpt:
                    for pred_val, prob in variable.cpt[parent_tuple_val].items():
                        if pred_val not in prediction_options:
                            prediction_options[pred_val] = 0
                        prediction_options[pred_val] += value_prob * prob

            if len(prediction_options) > 0:
                return prediction_options, False
            else:
                return {0: 0}, True
        else:
            list_val_orig = list(val_tuple)
            prediction_options = {}
            for idx in range(0, len(parents)):
                list_val = list_val_orig[:]
                curr_variable = parents[idx]
                for val in curr_variable.values:
                    list_val[idx] = val
                    new_tuple_val = tuple(list_val)
                    if new_tuple_val in variable.cpt:
                        for pred_val, prob in variable.cpt[new_tuple_val].items():
                            if pred_val not in prediction_options:
                                prediction_options[pred_val] = 0
                            prediction_options[pred_val] += curr_variable.values[val] * prob
            if len(prediction_options) > 0:
                return prediction_options, False
            else:
                return {0:0}, True


def predict_next_event_row(row, model, test_log):
    if getattr(row[1], test_log.activity + "_Prev" + str(test_log.k - 1)) == 0:
        return -1

    parents = model.variables[test_log.activity].conditional_parents

    value = []
    for parent in parents:
        value.append(getattr(row[1], parent.attr_name))
    tuple_val = tuple(value)

    probs, unknown = get_probabilities(model.variables[test_log.activity], tuple_val, parents)

    # Select value with highest probability
    predicted_val = max(probs, key=lambda l: probs[l])

    # Select value random according to probabilities
    """
    values = list(probs.keys())
    random_choice = random.random()
    values_idx = 0
    total_prob = probs[values[values_idx]]
    while total_prob < random_choice and values_idx < len(values) - 1:
        values_idx += 1
        total_prob += probs[values[values_idx]]
    predicted_val = values[values_idx]
    """

    if getattr(row[1], test_log.activity) == predicted_val:
        return 1
    else:
    #    if getattr(row[1], test_log.activity) in probs:
    #        print("Predicted prob:", probs[predicted_val], " Correct prob:", probs[getattr(row[1], test_log.activity)], probs)
    #    else:
    #        print("Unknown: not in probs")
        return 0


def predict_next_event(edbn_model, log):
    result = []

    with mp.Pool(mp.cpu_count()) as p:
        result = p.map(functools.partial(predict_next_event_row, model=edbn_model, test_log=log), log.contextdata.iterrows())

    result = [r for r in result if r != -1]
    correct = np.sum(result)
    false = len(result) - correct

    print(correct, false)
    print(correct / len(result))
    print(np.average(result))

def predict_suffix(model, test_log):
    all_parents, attributes = get_prediction_attributes(model, test_log.activity)

    END_EVENT = test_log.convert_string2int(test_log.activity, "END")

    predicted_cases = []
    prefix_results = {}
    unknown_values = 0
    total_predictions = 0

    for case_name, case in test_log.get_cases():
        case_events = case[test_log.activity].values
        for prefix_size in range(1, case.shape[0]): # Iterate over the different prefixes of the case
            total_predictions += 1
            # Create last known row (including known history, depending on k of the model)
            current_row = {}
            for iter_k in range(test_log.k, -1, -1):
                index = prefix_size - 1 - iter_k
                if index >= 0:
                    row = []
                    for attr in attributes:
                        row.append(getattr(case.iloc[index], attr))
                    current_row[iter_k] = row
                else:
                    current_row[iter_k] = [0] * len(attributes)

            # Predict suffix given the last known rows. Stop predicting when END_EVENT has been predicted or predicted size >= 100
            #predicted_rows, unknown_value = predict_case_suffix_highest_prob(all_parents, attributes, current_row, model, test_log.activity, END_EVENT)
            #predicted_rows, unknown_value = predict_case_suffix_random(all_parents, attributes, current_row, model, test_log.activity, END_EVENT, case[test_log.activity].values[prefix_size:])
            predicted_rows, unknown_value = predict_case_suffix_loop_threshold(all_parents, attributes, current_row, model, test_log.activity, END_EVENT)
            #predicted_rows, unknown_value = predict_case_suffix_return_end(all_parents, attributes, current_row, model, test_log.activity, END_EVENT)

            if unknown_value:
                unknown_values += 1

            # Get predicted trace
            predicted_events = [i[0] for i in predicted_rows]
            if prefix_size not in prefix_results:
                prefix_results[prefix_size] = []
            # Store similarity for predicted trace according to size of prefix
            prefix_results[prefix_size].append(1 - (damerau_levenshtein_distance(predicted_events, case_events[prefix_size:]) / max(len(predicted_events), len(case_events[prefix_size:]))))
            if prefix_size == 1:
                predicted_cases.append(predicted_events)
    avg_sims = []
    all_sims = []
    for prefix in sorted(prefix_results.keys()):
        print(prefix, np.average(prefix_results[prefix]))
        avg_sims.append(np.average(prefix_results[prefix]))
        all_sims.extend(prefix_results[prefix])
            #predicted_cases.append(predicted_rows)
    plt.plot(sorted(prefix_results.keys()), avg_sims)
    plt.show()
    #with mp.Pool(mp.cpu_count()) as p:
    #    sims = p.map(functools.partial(calculate_similarity, reference_logfile=test_log), predicted_cases)
    #print("Average Sim (DL - entire log):", np.average(sims))

    #print("Average Sim (leave out - DL):", np.average(calculate_similarity_leavout(predicted_cases, test_log)))

    print("Average Sim (DL - exact trace):", np.average(prefix_results[1]))

    print("Total Average Sim:", np.average(all_sims))
    print("Total unknown values:", unknown_values)
    print("Total predictions:", total_predictions )
    #print(sims)
        #print(case[attributes])
        #print(predicted_rows)
        #print("Case:", case_name, test_log.convert_int2string(test_log.trace, case_name))
        #for row in predicted_rows:
        #    print([test_log.convert_int2string(attributes[i], row[i]) for i in range(len(row))])

        #break


def predict_case_suffix_highest_prob(all_parents, attributes, current_row, model, activity_attr, end_event):
    """
    Predict the suffix for a case, given the latest known row(s)
    Selecting values with highest probability

    :param all_parents: detailed list of attributes
    :param attributes: ordered list of attributes
    :param current_row: current row, containg history
    :param model: eDBN model
    :param activity_attr: name of control flow attribute
    :param end_event: event indicating end of a trace
    :return: updated current_row
    """
    predicted_rows = []
    unknown_value = False

    while current_row[0][0] != end_event and len(predicted_rows) < 100:  # The event attribute should always be the first attribute in the list
        current_row[2] = current_row[1]
        current_row[1] = current_row[0]
        current_row[0] = [None] * len(all_parents)
        # Predict value for every attribute
        for attr in attributes:
            value = []
            for parent_detail in all_parents[attr]:
                value.append(current_row[parent_detail["k"]][attributes.index(parent_detail["name"])])
            tuple_val = tuple(value)

            probs, unknown = get_probabilities(model.variables[attr], tuple_val, [v["variable"] for v in all_parents[attr]])

            if unknown:
                unknown_value = True

            if 0 not in probs:
                prediction_value = max(probs, key=lambda l: probs[l])

                current_row[0][attributes.index(attr)] = prediction_value
            else:
                current_row[0][attributes.index(activity_attr)] = end_event
        predicted_rows.append(current_row[0][:])
    return predicted_rows, unknown_value


def predict_case_suffix_random(all_parents, attributes, current_row, model, activity_attr, end_event, correct_case = None):
    """
    Predict the suffix for a case, given the latest known row(s)
    Randomly choosing values according to their probabilities

    :param all_parents: detailed list of attributes
    :param attributes: ordered list of attributes
    :param current_row: current row, containg history
    :param model: eDBN model
    :param activity_attr: name of control flow attribute
    :param end_event: event indicating end of a trace
    :return: updated current_row
    """
    predicted_rows = []
    unknown_value = False

    while current_row[0][0] != end_event and len(predicted_rows) < 100:  # The event attribute should always be the first attribute in the list
        current_row[2] = current_row[1]
        current_row[1] = current_row[0]
        current_row[0] = [None] * len(all_parents)
        # Predict value for every attribute
        for attr in attributes:
            value = []
            for parent_detail in all_parents[attr]:
                value.append(current_row[parent_detail["k"]][attributes.index(parent_detail["name"])])
            tuple_val = tuple(value)

            probs, unknown = get_probabilities(model.variables[attr], tuple_val, [v["variable"] for v in all_parents[attr]])

            if unknown:
                unknown_value = True

            if 0 not in probs and len(probs) > 0:
                values = list(probs.keys())
                random_choice = random.random()
                values_idx = 0
                total_prob = probs[values[values_idx]]
                while total_prob < random_choice and values_idx < len(values) - 1:
                    values_idx += 1
                    total_prob += probs[values[values_idx]]

                current_row[0][attributes.index(attr)] = values[values_idx]
            else:
                current_row[0][attributes.index(activity_attr)] = end_event


        predicted_rows.append(current_row[0][:])
    return predicted_rows, unknown_value


def predict_case_suffix_loop_threshold(all_parents, attributes, current_row, model, activity_attr, end_event):
    """
    Predict the suffix for a case, given the latest known row(s)
    Selecting values with highest probability + Only allowing a limited amount of repetition of a single event

    :param all_parents: detailed list of attributes
    :param attributes: ordered list of attributes
    :param current_row: current row, containg history
    :param model: eDBN model
    :param activity_attr: name of control flow attribute
    :param end_event: event indicating end of a trace
    :return: updated current_row
    """
    predicted_rows = []
    repeated_event = [None]
    unknown_value = False

    while current_row[0][0] != end_event and len(predicted_rows) < 100:  # The event attribute should always be the first attribute in the list
        current_row[2] = current_row[1]
        current_row[1] = current_row[0]
        current_row[0] = [None] * len(all_parents)
        # Predict value for every attribute
        for attr in attributes:
            value = []
            for parent_detail in all_parents[attr]:
                value.append(current_row[parent_detail["k"]][attributes.index(parent_detail["name"])])
            tuple_val = tuple(value)

            probs, unknown = get_probabilities(model.variables[attr], tuple_val, [v["variable"] for v in all_parents[attr]])

            if unknown:
                unknown_value = True

            if 0 not in probs:
                max_val = None
                max_prob = 0
                for val, prob in probs.items():
                    duplicate_threshold = model.duplicate_events.get(val, 1)

                    if (prob > max_prob and attr != activity_attr) or \
                            (prob > max_prob and attr == activity_attr and repeated_event[0] != val) or \
                            (prob > max_prob and attr == activity_attr and len(repeated_event) <= duplicate_threshold):
                        max_prob = prob
                        max_val = val

                current_row[0][attributes.index(attr)] = max_val
            else:
                current_row[0][attributes.index(activity_attr)] = end_event

        if current_row[0][0] == repeated_event[0]:
            repeated_event.append(current_row[0][0])
        else:
            repeated_event = [current_row[0][0]]

        predicted_rows.append(current_row[0][:])
    return predicted_rows, unknown_value

def predict_case_suffix_return_end(all_parents, attributes, current_row, model, activity_attr, end_event):
    return [[end_event]], False

def get_prediction_attributes(model, activity_attribute):
    """
    Return lists containing attributes needed to predict in order to be able to predict the control flow

    :param model: eDBN model used
    :param activity_attribute:
    :return:
    """
    prev_pattern = re.compile(r"_Prev[0-9]*")
    all_parents = {}
    to_check = [activity_attribute]
    all_parents[activity_attribute] = model.variables[activity_attribute].conditional_parents[:]
    while len(to_check) > 0:
        attr = to_check[0]
        all_parents[attr] = model.variables[attr].conditional_parents[:]
        for parent in all_parents[attr]:
            current_attribute_version = prev_pattern.sub("", parent.attr_name)
            if current_attribute_version not in all_parents:
                to_check.append(current_attribute_version)
        to_check.remove(attr)
    print(all_parents)
    for par_attr in all_parents:
        detailed_attributes = []
        for parent in all_parents[par_attr]:
            attr_details = {}
            attr_details["name"] = prev_pattern.sub("", parent.attr_name)
            attr_details["variable"] = parent
            k = 0
            if "Prev" in parent.attr_name:
                attr_details["k"] = int(re.sub(r".*_Prev", "", parent.attr_name)) + 1
            else:
                attr_details["k"] = 0
            detailed_attributes.append(attr_details)
        all_parents[par_attr] = detailed_attributes
    attributes = list(all_parents.keys())
    return all_parents, attributes


def calculate_similarity(prediction, reference_logfile):
    temp_reference_log = reference_logfile.get_data() #.copy()
    cases = list(temp_reference_log.groupby([reference_logfile.trace]))
    min_sim = 1000
    min_index = 0
    for i in range(len(cases)):
        case_events = cases[i][1][reference_logfile.activity].values
        sim = damerau_levenshtein_distance(prediction, case_events) / (max(len(prediction), len(case_events)))
        if sim < min_sim:
            min_sim = sim
            min_index = i
    return 1 - min_sim

def calculate_similarity_leavout(predictions, reference_logfile):
    sims = []
    temp_reference_log = reference_logfile.get_data() #.copy()
    cases = list(temp_reference_log.groupby([reference_logfile.trace]))
    for prediction in predictions:
        min_sim = 1000
        min_index = 0
        for i in range(len(cases)):
            case_events = cases[i][1][reference_logfile.activity].values

            sim = damerau_levenshtein_distance(prediction, case_events) / (max(len(prediction), len(case_events)))
            if sim < min_sim:
                min_sim = sim
                min_index = i

        sims.append(1 - min_sim)
        del cases[min_index]
    return sims

"""
Compute the Damerau-Levenshtein distance between two given
lists (s1 and s2)
From: https://www.guyrutenberg.com/2008/12/15/damerau-levenshtein-distance-in-python/
"""
def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]


def learn_duplicated_events(logfile):
    duplicated_events = {}
    for case_name, case in logfile.get_cases():
        trace = case[logfile.activity].values
        current_count = 1
        prev_event = trace[0]
        for event_idx in range(1, len(trace)):
            if trace[event_idx] == prev_event:
                current_count += 1
            else:
                if current_count > 1:
                    if prev_event not in duplicated_events:
                        duplicated_events[prev_event] = []
                    duplicated_events[prev_event].append(current_count)
                prev_event = trace[event_idx]
                current_count = 1
    avg_duplicated_events = {}
    for event in duplicated_events:
        avg_duplicated_events[event] = int(np.average(duplicated_events[event]) + 1)
        #avg_duplicated_events[event] = max(duplicated_events[event])
    return avg_duplicated_events


def brier_multi(targets, probs):
    return np.mean(np.sum((probs - targets)**2))


def run_dataset():
    dataset = data.HELPDESK
    dataset_size = 2000000000
    add_end = True
    resource_pools =  False
    reduce_tasks = False
    logfile_k = 2
    bpic_file = 5

    remove_resource = True

    logfile, log_filename = data.get_data(dataset, dataset_size, logfile_k, add_end, reduce_tasks, resource_pools, remove_resource)
    #logfile.keep_attributes(["case", "event"])

    train_log, test_log = logfile.splitTrainTest(70)
    model = edbn.train(train_log)

    # Train average number of duplicated events
#    duplicates = learn_duplicated_events(train_log)
#    print(duplicates)
#    model.duplicate_events = duplicates

    predict_next_event(model, test_log)
    #predict_suffix(model, test_log)


def test_datasets():
    from LogFile import LogFile
    import pandas as pd
    from Utils.Uncertainty_Coefficient import calculate_mutual_information, calculate_entropy

    camargo_folder = "../Camargo/output_files/output_run3/BPIC15_20000000_2_1_0_0_1/shared_cat/data/"

    train_camargo = LogFile(camargo_folder + "train_log.csv", ",", 0, 20000000, None, "caseid",
                        activity_attr="task", convert=False, k=2)
    train_camargo.keep_attributes(["caseid", "task", "role"])

    test_camargo = LogFile(camargo_folder + "test_log.csv",
                       ",", 0, 20000000, None, "caseid",
                       activity_attr="task", convert=False, k=2, values=train_camargo.values)
    test_camargo.keep_attributes(["caseid", "task", "role"])

    total_camargo = pd.concat([test_camargo.data, train_camargo.data])

    total, name = data.get_data(data.BPIC15, 20000000, 2, True, False, False, True)
    total_data = total.data

    print(total_camargo)
    print(total_data)

    print(total_camargo.columns, total_data.columns)
    for idx in range(3):
        col1 = total_camargo[total_camargo.columns[idx]]
        col2 = total_data[total_data.columns[idx]]
        print(calculate_mutual_information(col1, col2), calculate_entropy(col1), calculate_entropy(col2))

if __name__ == "__main__":
    test_datasets()
    #run_dataset()

    """
    from LogFile import LogFile
    import os
    import pickle

    trainings = []
    trainings.append({"folder": "../Camargo/output_files/output_run3/BPIC12_20000000_2_1_0_0_1/shared_cat/data/", "model": "BPIC12_20000000_2_1_0_0_1"})
    trainings.append({"folder": "../Camargo/output_files/output_run3/BPIC12_20000000_2_1_0_1_1/shared_cat/data/", "model": "BPIC12_20000000_2_1_0_1_1"})
    #trainings.append({"folder": "../Camargo/output_files/output_run3/BPIC12W_20000000_2_1_0_0_1/shared_cat/data/", "model": "BPIC12W_20000000_2_1_0_0_1"})
    #trainings.append({"folder": "../Camargo/output_files/output_run3/BPIC12W_20000000_2_1_0_1_1/shared_cat/data/", "model": "BPIC12W_20000000_2_1_0_1_1"})
    #trainings.append({"folder": "../Camargo/output_files/output_run3/BPIC15_20000000_2_1_0_0_1/shared_cat/data/", "model": "BPIC15_20000000_2_1_0_0_1"})
    #trainings.append({"folder": "../Camargo/output_files/output_run3/BPIC15_20000000_2_1_0_1_1/shared_cat/data/", "model": "BPIC15_20000000_2_1_0_1_1"})
    #trainings.append({"folder": "../Camargo/output_files/output_run3/HELPDESK_20000000_2_1_0_0_1/shared_cat/data/", "model": "HELPDESK_20000000_2_1_0_0_1"})
    #trainings.append({"folder": "../Camargo/output_files/output_run3/HELPDESK_20000000_2_1_0_1_1/shared_cat/data/", "model": "HELPDESK_20000000_2_1_0_1_1"})

    for training in trainings:
        print("TRAIN:", training["model"])

        camargo_folder = training["folder"]
        model_file = training["model"]

        train_log = LogFile(camargo_folder + "train_log.csv", ",", 0, 20000000, None, "caseid",
                          activity_attr="task", convert=False, k=2)
        train_log.keep_attributes(["caseid", "task", "role"])

        model = None
        if os.path.exists(model_file):
            print("Reading model from file")
            with open(model_file, "rb") as pickle_file:
                model = pickle.load(pickle_file)
        else:
            print("Writing model to file")
            model = edbn.train(train_log)
            with open(model_file, "wb") as pickle_file:
                pickle.dump(model, pickle_file)

        test_log = LogFile(camargo_folder + "test_log.csv",
                        ",", 0, 20000000, None, "caseid",
                        activity_attr="task", convert=False, k=2, values=train_log.values)
        test_log.keep_attributes(["caseid", "task", "role"])

        test_log.create_k_context()
        predict_next_event(model, test_log)
        input("Done, waiting...")
        """