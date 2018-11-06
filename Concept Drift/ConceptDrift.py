import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from statsmodels import robust
import pickle
from scipy import stats

# Import LogFile for representing the data
from LogFile import LogFile

import eDBN.GenerateModel as gm

def create_model(training_structure_data, training_params_data, ignore_attrs=[]):
    """
    Create an eDBN model from the given data

    :param training_structure_data: data to be used to learn the structure of the model
    :param training_params_data: data to be used to learn the parameters of the model
    :param trace_attr: the attribute that indicates the trace
    :param ignore_attrs: attributes that have to be ignored when learning the model
    :return: the learned model
    """
    cbn = gm.generate_model(training_structure_data, 1, ignore_attrs, None, None)
    cbn.train_data(training_params_data)
    return cbn

def filter_attributes(data, filter_prefixes):
    """
    Remove attributes with the given prefixes from the data

    :param data: the original data
    :param filter_prefixes: a list of prefixes of attributes that should be removed from the data
    :return: the filtered data
    """
    remove_attrs = []
    for attr in data:
        for prefix in filter_prefixes:
            if attr.startswith(prefix):
                remove_attrs.append(attr)
                break
    return data.drop(remove_attrs, axis=1)


def get_event_scores(data, model):
    """
    Return the scores for all events grouped by trace

    :param data: data that has to be scored
    :param model: model to be used to score
    :return: all scores grouped by trace
    """
    return model.calculate_scores(data)

def get_event_detailed_scores(data, model):
    """
    Return the detailed decomposition of scores grouped by trace

    :param data: data that has to be scored
    :param model: model to be used to score
    :return: all detailed scores grouped by trace
    """
    return model.calculate_scores_detail(data)

def get_attribute_detailed_scores(data, model, attribute):
    """
    Return the detailed score for a single attribute

    :param data:
    :param model:
    :return:
    """


def plot_single_scores(scores):
    """
    Plot all accumulated trace scores

    :param scores: scores
    :return: None
    """
    def product(x):
        r = 1
        for e in x:
            r *= e
        return r

    y = []
    x = []
    for key in sorted(scores.keys()):
        #if product(scores[key]) != 0:
        #    y.append(math.log10(math.pow(product(scores[key]), 1/len(scores[key]))))
        #else:
        #    y.append(-10)

        if sum(scores[key]) != 0:
            y.append(math.log10(sum(scores[key]) / len(scores[key])))

        #for s in scores[key]:
        #    if s != 0:
        #        y.append(math.log10(s))
        #    else:
        #        y.append(-10)
        #    x.append(key)

    #plt.scatter(x,y)
    plt.scatter(range(len(y)), y)
    plt.xlabel("Traces")
    plt.ylabel("Log Scores")
    plt.show()

def plot_pvalues(scores, window):
    """
    Plot the pvalues for the given scores

    :param scores: the accumulated scores for all traces
    :param window: the window size to use
    :return: None
    """
    def createSlidingWindows(scores, window_size):
        windows = []
        for start in range(len(scores) - 2 * window_size):
            windows.append([scores[start:start + window_size], scores[start + window_size:start + 2 * window_size]])
        return windows

    case_scores = []
    for k in sorted(scores.keys()):
        if sum(scores[k]) != 0:
            case_scores.append(math.log10(sum(scores[k]) / len(scores[k])))
        else:
            case_scores.append(-50)

    windows = createSlidingWindows(case_scores, window)

    pvals = []
    for w in windows:
        pval = stats.ks_2samp(w[0], w[1]).pvalue
        if pval > 0:
            result = math.log10(pval)
        else:
            result = -350

        pvals.append(result)
    plt.plot(range(window, window + len(pvals)), pvals)
    plt.xlabel("Traces")
    plt.ylabel("Log p-values")
    plt.show()


def plot_attribute_graph(scores, attributes):
    """
    Plot all trace scores according to attribute

    :param scores: detailed scores
    :param attributes: attributes to plot
    :return: None
    """
    x_vals = []
    y_vals = []
    x_median = []
    y_median = []
    y_median_mad_up = []
    y_median_mad_down = []
    for a in sorted(attributes):
        for score in scores[a]:
            x_vals.append(a)
            y_vals.append(score)
        median = np.median(scores[a])
        mad = robust.mad(scores[a])
        x_median.append(a)
        y_median.append(median)
        y_median_mad_up.append(median + mad)
        y_median_mad_down.append(median - mad)

    plt.plot(x_vals, y_vals, "o")
    plt.plot(x_median, y_median, "o")
    plt.plot(x_median, y_median_mad_up, "o")
    plt.plot(x_median, y_median_mad_down, "o")
    plt.xticks(rotation='vertical')
    #plt.ylim([-12.1,0.1])
    plt.show()


def experiment_standard():
    data = LogFile("../Data/bpic2018.csv", ",", 0, 3000, "startTime", "case")
    #data_str = pd.read_csv("../Data/bpic2018_ints.csv", delimiter=",", header=0, dtype=int, nrows=3000)
    data.remove_attributes(["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    model = create_model(data, data)

    #with open("model_30000b", "wb") as fout:
    #    pickle.dump(model, fout)

    with open("model_30000b", "rb") as fin:
        model = pickle.load(fin)

    data = pd.read_csv("../Data/bpic2018_ints.csv", delimiter=",", header=0, dtype=int)
    data = filter_attributes(data, ["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    scores = get_event_scores(data, model)
    plot_single_scores(scores)
    plot_pvalues(scores, 800)

def experiment_attributes_standard():
    """
    data = pd.read_csv("../Data/bpic2018_ints.csv", delimiter=",", header=0, dtype=int, nrows=30000)
    data = filter_attributes(data, ["event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    model = create_model(data, data, "case")

    print("Starting writing model to file")
    with open("model_30000", "wb") as fout:
        pickle.dump(model, fout)
    print("Done")
    """
    with open("model_30000b", "rb") as fin:
        model = pickle.load(fin)

    input_data = pd.read_csv("../Data/bpic2018_ints.csv", delimiter=",", header=0, dtype=int)
    data = input_data[input_data.year == 1]
    data = filter_attributes(data, ["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    scores_year1 = get_event_detailed_scores(data, model)
    plot_attribute_graph(scores_year1, model.current_variables)

    data = input_data[input_data.year == 2]
    data = filter_attributes(data, ["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    scores_year2 = get_event_detailed_scores(data, model)
    plot_attribute_graph(scores_year2, model.current_variables)

    data = input_data[input_data.year == 3]
    data = filter_attributes(data, ["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    scores_year3 = get_event_detailed_scores(data, model)
    plot_attribute_graph(scores_year3, model.current_variables)

    p_vals_year1_2 = []
    p_vals_year2_3 = []
    p_vals_year1_3 = []
    for key in sorted(scores_year1.keys()):
        p_vals_year1_2.append(stats.ks_2samp(scores_year1[key], scores_year2[key]).pvalue)
        p_vals_year2_3.append(stats.ks_2samp(scores_year2[key], scores_year3[key]).pvalue)
        p_vals_year1_3.append(stats.ks_2samp(scores_year1[key], scores_year3[key]).pvalue)

    def tmp(x):
        if x == 0:
            return x
        else:
            return 1

    p_vals_year1_2 = [tmp(x) for x in p_vals_year1_2]
    plt.plot(sorted(scores_year1.keys()), p_vals_year1_2, "o")
    plt.xticks(rotation='vertical')
    plt.show()
    p_vals_year2_3 = [tmp(x) for x in p_vals_year2_3]
    plt.plot(sorted(scores_year1.keys()), p_vals_year2_3, "o")
    plt.xticks(rotation='vertical')
    plt.show()
    p_vals_year1_3 = [tmp(x) for x in p_vals_year1_3]
    plt.plot(sorted(scores_year1.keys()), p_vals_year1_3, "o")
    plt.xticks(rotation='vertical')
    plt.show()

    x = []
    y_1 = []
    y_2 = []
    y_3 = []
    for key in sorted(scores_year1.keys()):
        x.append(key)
        y_1.append(np.median(scores_year1[key]))
        y_2.append(np.median(scores_year2[key]))
        y_3.append(np.median(scores_year3[key]))
    plt.plot(x, y_1, "o")
    plt.plot(x, y_2, "o")
    plt.plot(x, y_3, "o")
    plt.xticks(rotation='vertical')
    plt.xlabel("Attributes")
    plt.ylabel("Median Score")
    plt.legend(["2015", "2016", "2017"])
    plt.show()

    p_vals_year1_2 = []
    p_vals_year2_3 = []
    p_vals_year1_3 = []
    for key in sorted(scores_year1.keys()):
        p_vals_year1_2.append(stats.ks_2samp(scores_year1[key], [np.median(scores_year1[key])]).pvalue)
        p_vals_year2_3.append(stats.ks_2samp(scores_year2[key], [np.median(scores_year2[key])]).pvalue)
        p_vals_year1_3.append(stats.ks_2samp(scores_year1[key], [np.median(scores_year3[key])]).pvalue)

    def tmp(x):
        if x == 0:
            return x
        else:
            return 1

    p_vals_year1_2 = [tmp(x) for x in p_vals_year1_2]
    plt.plot(sorted(scores_year1.keys()), p_vals_year1_2, "o")
    plt.xticks(rotation='vertical')
    plt.show()
    p_vals_year2_3 = [tmp(x) for x in p_vals_year2_3]
    plt.plot(sorted(scores_year1.keys()), p_vals_year2_3, "o")
    plt.xticks(rotation='vertical')
    plt.show()
    p_vals_year1_3 = [tmp(x) for x in p_vals_year1_3]
    plt.plot(sorted(scores_year1.keys()), p_vals_year1_3, "o")
    plt.xticks(rotation='vertical')
    plt.show()

def experiment_department():

    data = pd.read_csv("../Data/bpic2018_ints.csv", delimiter=",", header=0, dtype=int)
    data = data[data.department == 1]
    data = filter_attributes(data, ["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    model = create_model(data, data, "case")

    print("Starting writing model to file")
    with open("model_department", "wb") as fout:
        pickle.dump(model, fout)
    print("Done")

    with open("model_department", "rb") as fin:
        model = pickle.load(fin)

    input_data = pd.read_csv("../Data/bpic2018_ints.csv", delimiter=",", header=0, dtype=int)
    for dept in [1, 2, 3, 4]:
        data = input_data[input_data.department == dept]
        data = filter_attributes(data, ["event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
        scores = get_event_detailed_scores(data, model)
        plot_attribute_graph(scores, model.current_variables)

def experiment_clusters():
    with open("model_30000", "rb") as fin:
        model = pickle.load(fin)
    data = pd.read_csv("../Data/bpic2018_ints.csv", delimiter=",", header=0, dtype=int)
    data = data[data.year == 1]
    data = filter_attributes(data, ["event_identity_id", "year", "penalty_", "amount_applied", "payment_actual",
                                    "penalty_amount", "risk_factor", "cross_compliance", "selected_random",
                                    "selected_risk", "selected_manually", "rejected"])
    scores = get_event_detailed_scores(data, model)

    # First calculate score per trace
    attributes = list(scores.keys())
    num_traces = len(scores[attributes[0]])
    upper = {}
    lower = {}
    for a in attributes:
        upper[a] = []
        lower[a] = []

    for trace_ix in range(num_traces):
        score = 1
        for a in scores:
            a_score = scores[a][trace_ix]
            if a_score == -5:
                score = 0
                break
            score *= a_score

        if -8 < score < -10:
            for a in scores:
                upper[a].append(scores[a][trace_ix])
        elif -10 < score < -12:
            for a in scores:
                lower[a].append(scores[a][trace_ix])
    print(attributes)
    print(upper)
    plot_attribute_graph(upper, attributes)
    plot_attribute_graph(lower, attributes)

def experiment_outliers():
    with open("model_30000", "rb") as fin:
        model = pickle.load(fin)

    attr_dicts = []
    convert2ints("../Data/bpic2018.csv", "../Data/bpic2018_ints.csv", True, attr_dicts)


    data = pd.read_csv("../Data/bpic2018_ints.csv", delimiter=",", header=0, dtype=int)
    data = data[data.year == 1]
    data = filter_attributes(data, ["event_identity_id", "year", "penalty_", "amount_applied", "payment_actual",
                                    "penalty_amount", "risk_factor", "cross_compliance", "selected_random",
                                    "selected_risk", "selected_manually", "rejected"])

    scores = get_event_scores(data, model)
    for s in scores:
        if sum(scores[s]) != 0:
            score = math.log10(sum(scores[s]) / len(scores[s]))
            if score < -12:
                for case in attr_dicts[0]:
                    if attr_dicts[0][case] == s:
                        print(s, case, score)

def convert2ints(file_in, file_out, header = True, dict = None):
    cnt = 0
    with open(file_in, "r") as fin:
        with open(file_out, "w") as fout:
            if header:
                fout.write(fin.readline())
            for line in fin:
                cnt += 1
                input = line.replace("\n", "").split(",")
                if len(dict) == 0:
                    for t in range(len(input)):
                        dict.append({})
                output = []
                attr = 0
                for i in input:
                    if i not in dict[attr]:
                        dict[attr][i] = len(dict[attr]) + 1
                    output.append(str(dict[attr][i]))
                    attr += 1
                fout.write(",".join(output))
                fout.write("\n")
    return cnt

if __name__ == "__main__":
    experiment_standard()
    #experiment_attributes_standard()
    #experiment_department()
    #experiment_clusters()
    #experiment_outliers()
