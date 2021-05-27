import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels import robust

import Methods.EDBN.model.GenerateModel as gm


# Import LogFile for representing the data

def create_model(training_structure_data, training_params_data):
    """
    Create an EDBN model from the given data

    :param training_structure_data: data to be used to learn the structure of the model
    :param training_params_data: data to be used to learn the parameters of the model
    :param trace_attr: the attribute that indicates the trace
    :param ignore_attrs: attributes that have to be ignored when learning the model
    :return: the learned model
    """
    cbn = gm.generate_model(training_structure_data)
    cbn.train(training_params_data)
    return cbn


def get_event_scores(data, model):
    """
    Return the scores for all events grouped by trace

    :param data: data that has to be scored
    :param model: model to be used to score
    :return: all scores grouped by trace
    """
    return model.calculate_scores_per_trace(data)

def get_event_detailed_scores(data, model):
    """
    Return the detailed decomposition of scores grouped by trace

    :param data: data that has to be scored
    :param model: model to be used to score
    :return: all detailed scores grouped by trace
    """
    return model.calculate_scores_per_attribute(data)

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
    for trace_result in scores:

        score = trace_result.get_total_score() / trace_result.get_nr_events()
        if score != 0:
            y.append(score)

    plt.scatter(range(len(y)), y)
    plt.xlabel("Traces")
    plt.ylabel("Log Scores")
    plt.savefig("../Data/scores.png")
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
    for trace_score in scores:
        score = trace_score.get_total_score() / trace_score.get_nr_events()
        if score != 0:
            case_scores.append(score)
        else:
            case_scores.append(-5)

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
    plt.savefig("../Data/pvals.png")
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
        x_vals.extend([a] * len(scores[a]))
        y_vals.extend(scores[a])

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



