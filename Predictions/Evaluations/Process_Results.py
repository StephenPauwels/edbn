import Methods
from Predictions.Evaluations.experiments import get_scores, get_full_filename
import setting
import metric

from copy import copy


def process_standard():
    for dataset in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        print()
        for m in ["CAMARGO", "DBN", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "SDL", "TAX", "TAYMOURI"]:
            results = get_scores(dataset, m, setting.STANDARD)
            if results:
                acc = metric.ACCURACY.calculate(results)
            else:
                acc = 0
            print(acc, sep="\t")
        print()

def process_k():
    ks = [3, 5, 10]
    basic_setting = copy(setting.STANDARD)
    basic_setting.train_split = "k-fold"

    for dataset in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        print()
        for m in ["CAMARGO", "DBN", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "SDL", "TAX", "TAYMOURI"]:
            accs = []
            for k in ks:
                basic_setting.train_k = k
                results = get_scores(dataset, m, basic_setting)
                if results:
                    accs.append(metric.ACCURACY.calculate(results))
                else:
                    accs.append(0)
            print("\t\t".join([str(a) for a in accs]), sep="\t")
        print()


def process_split():
    splits = ["train-test", "test-train", "random", "k-fold"]
    basic_setting = copy(setting.STANDARD)

    for dataset in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        print()
        for m in ["CAMARGO", "DBN", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "SDL", "TAX", "TAYMOURI"]:
            accs = []
            for split in splits:
                basic_setting.train_split = split
                if split == "k-fold":
                    basic_setting.train_k = 3
                results = get_scores(dataset, m, basic_setting)
                if results:
                    accs.append(metric.ACCURACY.calculate(results))
                else:
                    accs.append(0)
            print("\t\t".join([str(a) for a in accs]), sep="\t")
        print()


def process_percentage():
    train_percentages = [60, 66, 70, 80]
    basic_setting = copy(setting.STANDARD)

    for dataset in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        print()
        for m in ["CAMARGO", "DBN", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "SDL", "TAX", "TAYMOURI"]:
            accs = []
            for percentage in train_percentages:
                basic_setting.train_percentage = percentage
                results = get_scores(dataset, m, basic_setting)
                if results:
                    accs.append(metric.ACCURACY.calculate(results))
                else:
                    accs.append(0)
            print("\t\t".join([str(a) for a in accs]), sep="\t")
        print()


def process_end_event():
    end_events = [True, False]
    basic_setting = copy(setting.STANDARD)

    for dataset in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        print()
        for m in ["CAMARGO", "DBN", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "SDL", "TAX", "TAYMOURI"]:
            accs = []
            for end_event in end_events:
                basic_setting.add_end = end_event
                results = get_scores(dataset, m, basic_setting)
                if results:
                    accs.append(metric.ACCURACY.calculate(results))
                else:
                    accs.append(0)
            print("\t\t".join([str(a) for a in accs]), sep="\t")
        print()


def process_split_cases():
    split_cases = [True, False]
    basic_setting = copy(setting.STANDARD)

    for dataset in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        print()
        for m in ["CAMARGO", "DBN", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "SDL", "TAX", "TAYMOURI"]:
            accs = []
            for split_case in split_cases:
                basic_setting.split_cases = split_case
                results = get_scores(dataset, m, basic_setting)
                if results:
                    accs.append(metric.ACCURACY.calculate(results))
                else:
                    accs.append(0)
            print("\t\t".join([str(a) for a in accs]))
        print()


def process_filter_cases():
    filters = [None, 5]
    basic_setting = copy(setting.STANDARD)

    for dataset in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        print()
        for m in ["CAMARGO", "DBN", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "SDL", "TAX", "TAYMOURI"]:
            accs = []
            for filter in filters:
                if filter == 5 and dataset == "Helpdesk":
                    filter = 3
                basic_setting.filter_cases = filter
                results = get_scores(dataset, m, basic_setting)
                if results:
                    accs.append(metric.ACCURACY.calculate(results))
                else:
                    accs.append(0)
            print("\t\t".join([str(a) for a in accs]))
        print()


def check_variance_k_folds(k_fold=3):
    check_setting = copy(setting.STANDARD)
    check_setting.train_split = "k-fold"
    check_setting.train_k = k_fold

    for m in ["CAMARGO", "DBN", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "SDL", "TAX", "TAYMOURI"]:
        results = get_scores("BPIC15_1", m, check_setting)
        length_folds = len(results) // k_fold
        accs = []
        for i in range(k_fold-1):
            accs.append(metric.ACCURACY.calculate(results[length_folds*i:length_folds*(i+1)]))
        accs.append(metric.ACCURACY.calculate(results[length_folds*(k_fold-1):]))
        accs.append(metric.ACCURACY.calculate(results))
        print("\t\t".join([str(a) for a in accs]))


def check_ranking():
    from setting import ALL as ALL_SETTINGS
    from Methods import ALL as ALL_METHODS
    from experiments import result_exists

    for d in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        for s in ALL_SETTINGS:
            for method in ALL_METHODS:
                if result_exists(d, method, s):
                    print("Exists")
                else:
                    print("Not Exists")


def process_ranking():
    import numpy as np
    from setting import ALL as ALL_SETTINGS
    from Methods import ALL as ALL_METHODS
    from experiments import result_exists

    missing = []
    table = {}
    for s in ALL_SETTINGS:
        for m in ALL_METHODS:
            if m == "DIMAURO":
                m = "Di Mauro"
            accs = []
            for d in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
                if result_exists(d, m, s):
                    try:
                        results = get_scores(d, m, s)
                        accs.append(metric.ACCURACY.calculate(results))
                    except:
                        print(get_full_filename(d, m, s))
                else:
                    missing.append((str(s),m,d))
            table[(s, m)] = np.mean(accs)

    res = {}
    for s in ALL_SETTINGS:
        i = 1
        for acc in [(k, table[k]) for k in sorted(table.keys(), key=lambda l: table[l], reverse=True) if k[0] == s]:
            if acc[0] not in res:
                res[acc[0]] = []
            res[acc[0]] = (i, acc[1])
            i += 1

    print(res)
    print()

    for m in ["CAMARGO", "DBN", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "SDL", "TAX", "TAYMOURI"]:
        line1 = []
        line2 = []
        for s in ALL_SETTINGS:
            result = res[(s,m)]
            if result[0] == 1:
                line1.append("\\textbf{%i}" % result[0])
                line2.append("\\textbf{\\emph{%.2f}}" % result[1])
            else:
                line1.append("%i" % result[0])
                line2.append("\\emph{%.2f}" % result[1])

        print("\\multirow{2}{*}{%s}" % m, "&", " & ".join(line1), "\\\\")
        print("&", " & ".join(line2), "\\\\")
        print("\\\\")

    print()
    print()
    print("MISSING")
    for m in missing:
        print(m)


def ranking_excel():
    from setting import ALL as ALL_SETTINGS
    from experiments import result_exists

    for data in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        for method in ["CAMARGO", "DBN", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "SDL", "TAX", "TAYMOURI"]:
            accs = []
            for setting in ALL_SETTINGS:
                if result_exists(data, method, setting):
                    try:
                        results = get_scores(data, method, setting)
                        accs.append(metric.ACCURACY.calculate(results))
                    except:
                        print(get_full_filename(data, method, setting))
            print("\t\t".join([str(a) for a in accs]))
        print()
        print()


if __name__ == "__main__":
    RESULT_FOLDER = "results"

    # check_ranking()
    process_ranking()
    ranking_excel()
    # check_result_files()
    # process_standard()
    # process_k()
    # process_split()
    # process_percentage()
    # process_end_event()
    # process_split_cases()
    # process_filter_cases()
    # check_variance_k_folds(3)