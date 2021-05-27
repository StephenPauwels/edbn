import setting

from copy import copy
import os

RESULT_FOLDER = "bise_results"

def get_filename(data, method, s):
    path = [RESULT_FOLDER]
    path.append(s.train_split)
    if s.train_split != "k-fold":
        path.append(str(s.train_percentage))
    else:
        path.append(str(s.train_k))
    if s.split_cases:
        path.append("event")
    else:
        path.append("case")
    if s.add_end:
        path.append("add_end")
    else:
        path.append("no_end")
    path.append(str(s.filter_cases))
    filename = method + "_" + data + ".csv"
    return path, filename


def get_full_filename(data, method, s):
    if isinstance(method, str):
        m_name = method
    else:
        m_name = method.name
    path, filename = get_filename(data, m_name, s)
    return "/".join(path) + "/" + filename


def result_exists(data, method, s):
    return os.path.exists(get_full_filename(data, method, s))

def check_result_files(data=None, methods=None):
    import Methods

    if data is None:
        data = ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]

    if methods is None:
        methods = Methods.ALL

    for d in data:
        print("DATASET: %s" % d)
        for m in methods:
            if m == "DIMAURO":
                m = "Di Mauro"
            print(" METHOD: %s" % m)
            # test_standard
            # test_k
            ks = [3, 5, 10]
            basic_setting = copy(setting.STANDARD)
            basic_setting.train_split = "k-fold"

            for k in ks:
                basic_setting.train_k = k
                print("  TEST K: %i %s" % (k, "OK" if result_exists(d, m, basic_setting) else ""))

            # test_split
            splits = ["train-test", "test-train", "random", "k-fold"]
            basic_setting = copy(setting.STANDARD)

            for split in splits:
                basic_setting.train_split = split
                print("  TEST SPLIT: %s %s" % (split, "OK" if result_exists(d, m, basic_setting) else ""))

            # test_filter
            filters = [None, 5]
            basic_setting = copy(setting.STANDARD)

            for filter in filters:
                basic_setting.filter_cases = filter
                if filter == 5 and d == "Helpdesk":
                    basic_setting.filter_cases = 3
                print("  TEST FILTER: %s %s" % (str(filter), "OK" if result_exists(d, m, basic_setting) else ""))

            # test_percentage
            train_percentages = [60, 66, 70, 80]
            basic_setting = copy(setting.STANDARD)

            for percentage in train_percentages:
                basic_setting.train_percentage = percentage
                print("  TEST PERCENTAGE: %s %s" % (percentage, "OK" if result_exists(d, m, basic_setting) else ""))

            # test_split_cases
            split_cases = [True, False]
            basic_setting = copy(setting.STANDARD)

            for split_case in split_cases:
                basic_setting.split_cases = split_case
                print("  TEST SPLIT CASES: %s %s" % ("True" if split_case else "False", "OK" if result_exists(d, m, basic_setting) else ""))

            # test_end_event
            end_events = [True, False]
            basic_setting = copy(setting.STANDARD)

            for end_event in end_events:
                basic_setting.add_end = end_event
                print("  TEST END EVENT: %s %s" % ("True" if end_event else "False", "OK" if result_exists(d, m, basic_setting) else ""))

def get_scores(d, m, s):
    path, filename = get_filename(d, m, s)
    if os.path.isfile("/".join(path) + "/" + filename):
        results = []
        with open("/".join(path) + "/" + filename) as finn:
            for line in finn:
                results.append(tuple(line.replace("\n", "").split(",")))

        import metric
        print("/".join(path) + "/" + filename)
        acc = metric.ACCURACY.calculate(results)
        print("ACC:", acc)
        prec = metric.PRECISION.calculate(results)
        print("Precision:", prec)
        recall = metric.RECALL.calculate(results)
        print("Recall:", recall)

if __name__ == "__main__":
    # check_result_files(methods=["Taymouri"])

    print(result_exists("Helpdesk", "Taymouri", setting.TAYMOURI))