import functools
import itertools
import multiprocessing as mp
import random
import re
from copy import deepcopy

import numpy as np

import EDBN.Execute as edbn
import Preprocessing as data

def cond_prob(a,b):
    return len(set.intersection(a,b)) / len(b)

def get_probabilities(variable, val_tuple, parents):
    if val_tuple in variable.cpt:
        return variable.cpt[val_tuple], False
    else:
        predictions = {}
        unseen_value = False
        value_combinations = []
        known_attributes_indexes = None
        unseen_attribute_i = []
        for i in range(len(val_tuple)):
            if val_tuple[i] not in parents[i].value_counts:
                unseen_value = True
                value_combinations.append(parents[i].value_counts.keys())
                unseen_attribute_i.append(i)
            else:
                value_combinations.append([val_tuple[i]])
                if known_attributes_indexes is None:
                    known_attributes_indexes = parents[i].value_counts[val_tuple[i]]
                else:
                    known_attributes_indexes = set.intersection(known_attributes_indexes, parents[i].value_counts[val_tuple[i]])

        if unseen_value:
            for combination in itertools.product(*value_combinations):
                if combination not in variable.cpt:
                    continue

                unseen_attributes = [parents[i].value_counts[combination[i]] for i in unseen_attribute_i]
                unseen_indexes = set.intersection(*unseen_attributes)

                parent_prob = cond_prob(unseen_indexes, known_attributes_indexes)
                parent_indexes = set.intersection(unseen_indexes, known_attributes_indexes)

                for value in variable.cpt[combination]:
                    if value not in predictions:
                        predictions[value] = 0
                    predictions[value] += variable.cpt[combination][value] * parent_prob

            if len(predictions) > 0:
                return predictions, True
            else:
                return {0: 0}, True
        else:
            #Unseen value combination
            for i in range(len(val_tuple)):
                values = [[v] for v in val_tuple]
                attr = parents[i]
                values[i] = attr.value_counts.keys()
                for combination in itertools.product(*values):
                    if combination not in variable.cpt:
                        continue

                    fixed_attrs = [parents[j].value_counts[combination[j]] for j in range(len(combination)) if j != i]
                    fixed_indexes = set.intersection(*fixed_attrs)

                    variable_indexes = attr.value_counts[combination[i]]

                    parent_prob = cond_prob(variable_indexes, fixed_indexes)

                    parents_indexes = set.intersection(fixed_indexes, variable_indexes)

                    for value in variable.cpt[combination]:
                        if value not in predictions:
                            predictions[value] = 0
                        predictions[value] += variable.cpt[combination][value] * parent_prob

            for pred_val in predictions:
                predictions[pred_val] = predictions[pred_val] / len(parents)

            if len(predictions) > 0:
                return predictions, True
            else:
                return {0: 0}, True

        return {0: 0}, True


def get_probabilities_old(variable, val_tuple, parents):
    if val_tuple in variable.cpt:
        return variable.cpt[val_tuple], False
    else:
        #return {0:0}, True
        # Check if single values occur in dataset
        value_probs = []
        for i in range(len(val_tuple)):
            if val_tuple[i] not in parents[i].values:
                value_probs.append(None)
            else:
                value_probs.append(parents[i].values[val_tuple[i]])

        if None in value_probs:
            possible_values = []
            # Iterate over all known parent configurations
            for par_config in variable.cpt:
                match_config = True
                config_prob = 1
                for i in range(len(par_config)):
                    if value_probs[i] is None:
                        config_prob *= parents[i].values[par_config[i]]
                    elif value_probs[i] != par_config[i]:
                        match_config = False
                        break
                if match_config:
                    possible_values.append((par_config, config_prob))

            prediction_options = {}
            for parent_value, parent_prob in possible_values:
                for pred_val, prob in variable.cpt[parent_value].items():
                    if pred_val not in prediction_options:
                        prediction_options[pred_val] = 0
                    prediction_options[pred_val] += parent_prob * prob

            if len(prediction_options) > 0:
                return prediction_options, True
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
                            prediction_options[pred_val] += variable.cpt_probs[new_tuple_val] * prob
            if len(prediction_options) > 0:
                return prediction_options, True
            else:
                return {0:0}, True


def predict_next_event_row(row, model, test_log):
    parents = model.variables[test_log.activity].conditional_parents

    value = []
    for parent in parents:
        value.append(getattr(row[1], parent.attr_name))
    tuple_val = tuple(value)

    # TODO: Check if first event

    activity_var = model.variables[test_log.activity]
    probs, unknown = get_probabilities(activity_var, tuple_val, parents)

    # Select value with highest probability
    if not unknown:
        predicted_val = max(probs, key=lambda l: probs[l] * activity_var.values[l])
    else:
        predicted_val = max(probs, key=lambda l: probs[l])

    if getattr(row[1], test_log.activity) == predicted_val:
        return 1
    else:
        return 0



def predict_next_event(edbn_model, log):
    result = []

    with mp.Pool(mp.cpu_count()) as p:
        result = p.map(functools.partial(predict_next_event_row, model=edbn_model, test_log=log), log.contextdata.iterrows())

    # with open("results_next_event.csv", "w") as fout:
    #     for r in result:
    #         if r[1] is not None:
    #             fout.write(",".join(['"' +log.convert_int2string("trace", r[1]).replace("'", "") + '"', str(r[0])]))
    #             fout.write("\n")

    # result = [r[0] for r in result if r[0] != -1]
    result = [r for r in result if r != -1]
    correct = np.sum(result)
    false = len(result) - correct

    print(correct, false)
    print(correct / len(result))
    print(np.average(result))
    return np.average(result)

def predict_suffix(model, test_log):
    all_parents, attributes = get_prediction_attributes(model, test_log.activity)

    prefix_results = {}
    total_predictions = 0

    predict_case_func = functools.partial(predict_case, all_parents=all_parents, attributes=attributes, model=model,
                                          end_event=test_log.convert_string2int(test_log.activity, "END"),
                                          activity_attr=test_log.activity, k=test_log.k)
    with mp.Pool(mp.cpu_count()) as p:
        results = p.map(predict_case_func, test_log.get_cases())

    #results = []
    #for case in test_log.get_cases():
    #    results.append(predict_case(case, all_parents, attributes, model, test_log.convert_string2int(test_log.activity, "END"), test_log.activity ))

    for result in results:
        for prefix_size in result:
            if prefix_size not in prefix_results:
                prefix_results[prefix_size] = []
            prefix_results[prefix_size].extend(result[prefix_size])

    avg_sims = []
    all_sims = []
    for prefix in sorted(prefix_results.keys()):
        avg_sims.append(np.average(prefix_results[prefix]))
        all_sims.extend(prefix_results[prefix])
            #predicted_cases.append(predicted_rows)
    #plt.plot(sorted(prefix_results.keys()), avg_sims)
    #plt.show()
    #with mp.Pool(mp.cpu_count()) as p:
    #    sims = p.map(functools.partial(calculate_similarity, reference_logfile=test_log), predicted_cases)
    #print("Average Sim (DL - entire log):", np.average(sims))

    #print("Average Sim (leave out - DL):", np.average(calculate_similarity_leavout(predicted_cases, test_log)))

    print("Average Sim (DL - exact trace):", np.average(prefix_results[1]))

    total_sim = np.average(all_sims)
    print("Total Average Sim:", total_sim)
    print("Total predictions:", total_predictions )
    #print(sims)
        #print(case[attributes])
        #print(predicted_rows)
        #print("Case:", case_name, test_log.convert_int2string(test_log.trace, case_name))
        #for row in predicted_rows:
        #    print([test_log.convert_int2string(attributes[i], row[i]) for i in range(len(row))])

        #break
    return total_sim


def predict_case(case, all_parents, attributes, model, end_event, activity_attr, k):
    prefix_results = {}
    case = case[1]
    case_events = case[activity_attr].values
    for prefix_size in range(1, case.shape[0]):  # Iterate over the different prefixes of the case
        # Create last known row (including known history, depending on k of the model)
        current_row = {}
        for iter_k in range(k, -1, -1):
            index = prefix_size - 1 - iter_k
            if index >= 0:
                row = []
                for attr in attributes:
                    row.append(getattr(case.iloc[index], attr))
                current_row[iter_k] = row
            else:
                current_row[iter_k] = [0] * len(attributes)

        # Predict suffix given the last known rows. Stop predicting when END_EVENT has been predicted or predicted size >= 100
        #predicted_rows, unknown_value = predict_case_suffix_highest_prob(all_parents, attributes, current_row, model, activity_attr, end_event)
        # predicted_rows, unknown_value = predict_case_suffix_random(all_parents, attributes, current_row, model, test_log.activity, end_event)
        predicted_rows, unknown_value = predict_case_suffix_loop_threshold(all_parents, attributes, current_row, model,activity_attr, end_event)
        # predicted_rows, unknown_value = predict_case_suffix_return_end(all_parents, attributes, current_row, model, test_log.activity, END_EVENT)

        # Get predicted trace
        predicted_events = [i[0] for i in predicted_rows if i[0] is not None]
        if prefix_size not in prefix_results:
            prefix_results[prefix_size] = []
        # Store similarity for predicted trace according to size of prefix
        prefix_results[prefix_size].append(1 - (
                    damerau_levenshtein_distance(predicted_events, case_events[prefix_size:]) / max(
                len(predicted_events), len(case_events[prefix_size:]))))
    return prefix_results


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

def run_dataset(dataset = None, k = 2):
    from datetime import datetime

    if dataset is None:
        dataset = data.BPIC15_1
    dataset_size = 200000000
    add_end = False
    resource_pools = False
    reduce_tasks = False
    logfile_k = k
    bpic_file = 5

    remove_resource = False

    logfile, log_filename = data.get_data(dataset, dataset_size, logfile_k, add_end, reduce_tasks, resource_pools, remove_resource)
    logfile.convert2int()
    train_log, test_log = logfile.splitTrainTest(70)
    train_log.create_k_context()
    test_log.create_k_context()

    model = edbn.train(train_log)

    # Train average number of duplicated events
#    print(duplicates)
    model.duplicate_events = learn_duplicated_events(train_log)
    acc = predict_next_event(model, test_log)
    with open("Results.csv", "a") as fout:
        fout.write(",".join([str(datetime.now()), dataset + "_" + str(logfile_k), "Predict_Next", str(acc)]))
        fout.write("\n")

    # sim =predict_suffix(model, test_log)
    # with open("Results.csv", "a") as fout:
    #     fout.write(",".join([str(datetime.now()), dataset + "_" + str(logfile_k), "Predict_Suffix", str(sim)]))
    #     fout.write("\n")

def test_datasets():
    from LogFile import LogFile
    import pandas as pd
    from Utils.Uncertainty_Coefficient import calculate_mutual_information, calculate_entropy

    camargo_folder = "../Camargo/output_files/output_run3/BPIC12_20000000_2_1_0_0_1/shared_cat/data/"

    train_camargo = LogFile(camargo_folder + "train_log.csv", ",", 0, 20000000, None, "caseid",
                        activity_attr="task", convert=False, k=2)
    train_camargo.keep_attributes(["caseid", "task", "role"])

    test_camargo = LogFile(camargo_folder + "test_log.csv",
                       ",", 0, 20000000, None, "caseid",
                       activity_attr="task", convert=False, k=2, values=train_camargo.values)
    test_camargo.keep_attributes(["caseid", "task", "role"])

    total_camargo = pd.concat([test_camargo.data, train_camargo.data])

    total, name = data.get_data(data.BPIC12, 20000000, 2, True, False, False, True)
    total_data = total.data

    print(total_camargo)
    print(total_data)

    print(total_camargo.columns, total_data.columns)
    for idx in range(3):
        col1 = total_camargo[total_camargo.columns[idx]]
        col2 = total_data[total_data.columns[idx]]
        print(calculate_mutual_information(col1, col2), calculate_entropy(col1), calculate_entropy(col2))

def run_Tax():
    from LogFile import LogFile

    logfile = LogFile("../Tax/data/helpdesk.csv", ",", 0, 20000000, None, "CaseID", activity_attr="ActivityID", convert=True, k=2)
    logfile.keep_attributes(["CaseID", "ActivityID"])
    logfile.create_k_context()

    train_log, test_log = logfile.splitTrainTest(70)
    model = edbn.train(train_log)

    # Train average number of duplicated events
    model.duplicate_events = learn_duplicated_events(train_log)

    predict_next_event(model, test_log)
    #predict_suffix(model, test_log)


if __name__ == "__main__":
    #test_datasets()
    #run_dataset(data.BPIC15_1, 4)
    # for k in [2]: #[1,2,3,4,5]:
    #     for dataset in [data.BPIC12, data.BPIC12W, data.BPIC15_1, data.BPIC15_2, data.BPIC15_3, data.BPIC15_4, data.BPIC15_5, data.HELPDESK]:
    #        run_dataset(dataset, k)
    #run_Tax()


    from LogFile import LogFile
    from datetime import datetime
    import os
    import pickle

    trainings = []
    trainings.append({"folder": "../Data/PredictionData/bpic15_1/", "model": "BPIC15_1"})
    # trainings.append({"folder": "../Camargo/output_files/data/bpic15_1/", "model": "BPIC15_1"})
    # trainings.append({"folder": "../Camargo/output_files/data/bpic15_2/", "model": "BPIC15_2"})
    # trainings.append({"folder": "../Camargo/output_files/data/bpic15_3/", "model": "BPIC15_3"})
    # trainings.append({"folder": "../Camargo/output_files/data/bpic15_4/", "model": "BPIC15_4"})
    # trainings.append({"folder": "../Camargo/output_files/data/bpic15_5/", "model": "BPIC15_5"})
    # trainings.append({"folder": "../Camargo/output_files/data/bpic12W/", "model": "BPIC12W"})
    # trainings.append({"folder": "../Camargo/output_files/data/bpic12/", "model": "BPIC12"})
    # trainings.append({"folder": "../Camargo/output_files/data/helpdesk/", "model": "HELPDESK"})



    if not os.path.exists("Results.csv"):
        with open("Results.csv", "w") as fout:
            fout.write("Date,Model,Type,Average Similarity")
            fout.write("\n")

    for training in trainings:
        print("TRAIN:", training["model"])

        camargo_folder = training["folder"]
        model_file = training["model"]

        train_log = LogFile(camargo_folder + "train_log.csv", ",", 0, 20000000, None, "case",
                          activity_attr="event", convert=False, k=4)
        # train_log.create_trace_attribute()
        train_log.keep_attributes(["case", "event", "role"])
        #train_log.add_end_events()
        train_log.convert2int()

        train_log.create_k_context()

        model = None
        if os.path.exists(model_file):
            print("Reading model from file")
            with open(model_file, "rb") as pickle_file:
                model = pickle.load(pickle_file)
            model.print_parents()
        else:
            print("Writing model to file")
            model = edbn.train(train_log)

            # Train average number of duplicated events
            model.duplicate_events = learn_duplicated_events(train_log)

            with open(model_file, "wb") as pickle_file:
                pickle.dump(model, pickle_file)

        test_log = LogFile(camargo_folder + "test_log.csv",
                        ",", 0, 20000000, None, "case",
                        activity_attr="event", convert=False, k=4, values=train_log.values)
        # test_log.create_trace_attribute()
        test_log.keep_attributes(["case", "event", "role"])
        #test_log.add_end_events()
        test_log.convert2int()
        test_log.create_k_context()


        acc = predict_next_event(model, test_log)
        with open("Results.csv", "a") as fout:
            fout.write(",".join([str(datetime.now()), model_file, "Predict_Next", str(acc)]))
            fout.write("\n")



        # sim = predict_suffix(model, test_log)
        # with open("Results.csv", "a") as fout:
        #     fout.write(",".join([str(datetime.now()), model_file, "Predict_Suffix", str(sim)]))
        #     fout.write("\n")

