import matplotlib.pyplot as plt
from sklearn.metrics import auc

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

def plot_single_prec_recall_curve(result_file, prec_recall=None, save_file=None):
    precision, recall = calc_prec_recall(read_file(result_file))
    prec_recall_auc = auc(recall, precision)

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='Precision-Recall curve (area = %0.2f)' % prec_recall_auc)
    print("Prec-Recall auc:", prec_recall_auc)
    if prec_recall:
        plt.plot([prec_recall[1]], [prec_recall[0]], marker='o', markersize=3, color="red")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision - Recall Curve')
    plt.legend(loc="lower right")
    if save_file:
        plt.savefig(save_file)
    plt.show()

def plot_single_roc_curve(result_file, save_file=None):
    fpr, tpr = calc_roc(read_file(result_file))
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    print("ROC auc:", roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if save_file:
        plt.savefig(save_file)
    plt.show()

def get_roc_auc(result_file):
    fpr, tpr = calc_roc(read_file(result_file))
    return auc(fpr, tpr)

def plot_compare_prec_recall_curve(result_files, labels, save_file=None):
    prec_recall_vals = []
    auc_vals = []
    for file in result_files:
        prec_recall_vals.append(calc_prec_recall(read_file(file)))
        auc_vals.append(auc(prec_recall_vals[-1][1], prec_recall_vals[-1][0]))

    plt.figure()
    lw = 3
    for i in range(len(prec_recall_vals)):
        plt.plot(prec_recall_vals[i][1], prec_recall_vals[i][0],
             lw=lw, label='%s (area = %0.2f)' % (labels[i], auc_vals[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision - Recall Curve')
    plt.legend(loc="lower right")
    if save_file:
        plt.savefig(save_file)
    plt.show()

def plot_compare_roc_curve(result_files, labels, save_file=None):
    roc_vals = []
    auc_vals = []
    for file in result_files:
        roc_vals.append(calc_roc(read_file(file)))
        auc_vals.append(auc(roc_vals[-1][0], roc_vals[-1][1]))

    plt.figure()
    lw = 3
    for i in range(len(roc_vals)):
        plt.plot(roc_vals[i][0], roc_vals[i][1],
             lw=lw, label='%s (area = %0.2f)' % (labels[i], auc_vals[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    if save_file:
        plt.savefig(save_file)
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