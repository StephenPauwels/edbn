"""
    Author: Stephen Pauwels
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

def read_file(file):
    result = []
    with open(file) as fin:
        for line in fin:
            split_line = line.split(",")
            result.append((int(split_line[0]), float(split_line[1]), 0, eval(split_line[2])))
    return result

def calc(results):
    true_vals = []
    found_vals = []
    for result in results:
        if result[3]:
            true_vals.append(0)
        else:
            true_vals.append(1)
        found_vals.append(result[1])
    return true_vals, found_vals

def plot_single_prec_recall_curve(result_file, title=None, prec_recall=None, save_file=None):
    precision, recall = calc_prec_recall(read_file(result_file))
    prec_recall_auc = auc(recall, precision)

    plt.figure()

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.yticks(np.linspace(0,1,11), ["%0.1f" % x for x in np.linspace(0,1,11)], fontsize=14)
    plt.xticks(fontsize=14)

    for y in np.linspace(0,1,11):
        plt.plot(np.linspace(0,1,11), [y] * len(np.linspace(0,1,11)), "--", lw=0.5, color="black", alpha=0.3)

    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    labelbottom=True, left=False, right=False, labelleft=True)

    plt.plot(recall, precision, color='darkorange',
             lw=2, label='Precision-Recall curve (area = %0.2f)' % prec_recall_auc)
    print("EVALUATION: AUC PR:", prec_recall_auc)
    if prec_recall:
        plt.plot([prec_recall[1]], [prec_recall[0]], marker='o', markersize=3, color="red")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision - Recall Curve')
    if title:
        plt.title(title + " (Precision - Recall Curve)")
    else:
        plt.title('Precision - Recall Curve')
    plt.legend(loc="lower right")
    if save_file:
        plt.savefig(save_file)
    plt.show()

def plot_single_roc_curve(result_file, title=None, save_file=None):
    fpr, tpr = calc_roc(read_file(result_file))
    roc_auc = auc(fpr, tpr)

    plt.figure()

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.yticks(np.linspace(0,1,11), ["%0.1f" % x for x in np.linspace(0,1,11)], fontsize=14)
    plt.xticks(fontsize=14)

    for y in np.linspace(0,1,11):
        plt.plot(np.linspace(0,1,11), [y] * len(np.linspace(0,1,11)), "--", lw=0.5, color="black", alpha=0.3)

    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    labelbottom=True, left=False, right=False, labelleft=True)

    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    print("EVALUATION: AUC ROC:", roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title:
        plt.title(title + " (Receiver operating characteristic)")
    else:
        plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if save_file:
        plt.savefig(save_file)
    plt.show()

def get_roc_auc(result_file):
    fpr, tpr = calc_roc(read_file(result_file))
    return auc(fpr, tpr)

def plot_compare_prec_recall_curve(result_files, labels, prec_recalls=None, title=None, save_file=None):
    prec_recall_vals = []
    auc_vals = []
    for file in result_files:
        prec_recall_vals.append(calc_prec_recall(read_file(file)))
        auc_vals.append(auc(prec_recall_vals[-1][1], prec_recall_vals[-1][0]))

    plt.figure()

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.yticks(np.linspace(0,1,11), ["%0.1f" % x for x in np.linspace(0,1,11)], fontsize=14)
    plt.xticks(fontsize=14)

    for y in np.linspace(0,1,11):
        plt.plot(np.linspace(0,1,11), [y] * len(np.linspace(0,1,11)), "--", lw=0.5, color="black", alpha=0.3)

    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    labelbottom=True, left=False, right=False, labelleft=True)

    y = 0
    for i in range(len(prec_recall_vals)):
        plt.plot(prec_recall_vals[i][1], prec_recall_vals[i][0],
             lw=2, label='%s (area = %0.2f)' % (labels[i], auc_vals[i]), color=tableau20[i])
        y += 1

    if prec_recalls:
        for prec_recall in prec_recalls:
            plt.plot([prec_recall[1]], [prec_recall[0]], label=labels[y], marker='o', markersize=5, color=tableau20[y])
            y += 1

    print("EVALUATION: AUC PR :", labels, auc_vals)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if title:
        plt.title(title + " (Precision - Recall Curve)")
    else:
        plt.title('Precision - Recall Curve')
    plt.legend(loc="lower right")
    if save_file:
        plt.savefig(save_file)
    plt.show()

def plot_compare_roc_curve(result_files, labels, title=None, save_file=None):
    roc_vals = []
    auc_vals = []
    for file in result_files:
        roc_vals.append(calc_roc(read_file(file)))
        auc_vals.append(auc(roc_vals[-1][0], roc_vals[-1][1]))

    plt.figure()

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.yticks(np.linspace(0,1,11), ["%0.1f" % x for x in np.linspace(0,1,11)], fontsize=14)
    plt.xticks(fontsize=14)

    for y in np.linspace(0,1,11):
        plt.plot(np.linspace(0,1,11), [y] * len(np.linspace(0,1,11)), "--", lw=0.5, color="black", alpha=0.3)

    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    labelbottom=True, left=False, right=False, labelleft=True)

    for i in range(len(roc_vals)):
        plt.plot(roc_vals[i][0], roc_vals[i][1],
             lw=2, label='%s (area = %0.2f)' % (labels[i], auc_vals[i]), color=tableau20[i])

    print("EVALUATION: AUC ROC:", labels, auc_vals)

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    if title:
        plt.title(title + " (Receiver operating characteristic)")
    else:
        plt.title('ROC Curve')
    plt.legend(loc="lower right")
    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()

# Input: (Case ID, score, Anom)
def calc_prec_recall(values):
    total_pos = 0
    for v in values:
        if v[3]:
            total_pos += 1
    precision = []
    recall = []
    true_pos = 0
    true_neg = 0
    total = len(values)
    i = 0
    for v in sorted(values, key=lambda l: l[1]):
        i += 1
        if v[3]:
            true_pos += 1
        else:
            true_neg += 1
        false_pos = i - true_pos
        false_neg = total_pos - true_pos
        prec = true_pos / (true_pos + false_pos)
        rec = true_pos / (true_pos + false_neg)
        precision.append(prec)
        recall.append(rec)
    return precision, recall

def calc_roc(values):
    total_pos = 0
    total_neg = 0
    for v in values:
        if not v[3]:
            total_neg += 1
        else:
            total_pos += 1
    tprs = []
    fprs = []
    true_pos = 0
    false_pos = 0
    i = 0
    for v in sorted(values, key=lambda l: l[1]):
        i += 1
        if v[3]:
            true_pos += 1
        else:
            false_pos += 1
        true_neg = 0
        false_neg = 0
        fpr = false_pos / total_neg
        tpr = true_pos / total_pos
        fprs.append(fpr)
        tprs.append(tpr)
    return fprs, tprs

def calc_prec_recall_f1(file):
    results = read_file(file)

    true_pos = 0
    false_pos = 0
    false_neg = 0
    for result in results:
        if result[1] == 0 and result[3]:
            true_pos += 1
        elif result[1] == 0 and not result[3]:
            false_pos += 1
        elif result[3]:
            false_neg += 1

    print("True Pos:", true_pos)
    print("False Pos:", false_pos)
    print("False neg:", false_neg)

    prec = true_pos / (true_pos + false_pos)
    rec = true_pos / (true_pos + false_neg)
    print("PRECISION:", prec)
    print("RECALL:", rec)
    print("F1:", 2 * (prec * rec) / (prec + rec))
    return 2 * (prec * rec) / (prec + rec)