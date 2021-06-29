import traceback
from copy import copy
import os

from Data import get_data
from Methods import ALL as ALL_METHODS
import Predictions.setting as setting
from Predictions.setting import ALL as ALL_SETTINGS
from Predictions.metric import ACCURACY


RESULT_FOLDER = "results"


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


def save_results(results, data, method, setting):
    path, filename = get_filename(data, method, setting)
    for i in range(1, len(path)+1):
        if not os.path.exists("/".join(path[:i])):
            os.mkdir("/".join(path[:i]))

    with open("/".join(path) + "/" + filename, "w") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")


def test_split(dataset, m):
    splits = ["train-test", "test-train", "random", "k-fold"]
    basic_setting = copy(setting.STANDARD)

    for split in splits:
        d = get_data(dataset)
        basic_setting.train_split = split
        if split == "k-fold":
            basic_setting.train_k = 3

        print(get_full_filename(dataset, m, basic_setting))
        if result_exists(dataset, m, basic_setting):
            continue

        d.prepare(basic_setting)

        if split == "k-fold":
            r = m.k_fold_validation(d)
        else:
            r = m.test(m.train(d.train), d.test_orig)

        save_results(r, d.name, m.name, basic_setting)


def test_k(dataset, m):
    ks = [10]
    basic_setting = copy(setting.STANDARD)
    basic_setting.train_split = "k-fold"

    for k in ks:
        d = get_data(dataset)
        basic_setting.train_k = k

        print(get_full_filename(dataset, m, basic_setting))
        if result_exists(dataset, m, basic_setting):
            continue

        d.prepare(basic_setting)
        try:
            r = m.k_fold_validation(d)
        except:
            pass

        save_results(r, d.name, m.name, basic_setting)


def test_percentage(dataset, m):
    train_percentages = [60, 66, 70, 80]
    basic_setting = copy(setting.STANDARD)

    for percentage in train_percentages:
        d = get_data(dataset)
        basic_setting.train_percentage = percentage

        print(get_full_filename(dataset,m, basic_setting))
        if result_exists(dataset, m, basic_setting):
            continue

        d.prepare(basic_setting)

        r = m.test(m.train(d.train), d.test_orig)

        save_results(r, d.name, m.name, basic_setting)


def test_end_event(dataset, m):
    end_events = [True, False]
    basic_setting = copy(setting.STANDARD)

    for end_event in end_events:
        d = get_data(dataset)
        basic_setting.add_end = end_event

        print(get_full_filename(dataset,m, basic_setting))
        if result_exists(dataset, m, basic_setting):
            continue

        d.prepare(basic_setting)

        r = m.test(m.train(d.train), d.test_orig)

        save_results(r, d.name, m.name, basic_setting)


def test_split_cases(dataset, m):
    split_cases = [True, False]
    basic_setting = copy(setting.STANDARD)

    for split_case in split_cases:
        d = get_data(dataset)
        basic_setting.split_cases = split_case

        print(get_full_filename(dataset,m, basic_setting))
        if result_exists(dataset, m, basic_setting):
            continue

        d.prepare(basic_setting)

        r = m.test(m.train(d.train), d.test_orig)

        save_results(r, d.name, m.name, basic_setting)


def test_filter(dataset, m):
    filters = [None, 5]
    basic_setting = copy(setting.STANDARD)

    for filter in filters:
        d = get_data(dataset)
        basic_setting.filter_cases = filter
        if filter == 5 and dataset == "Helpdesk":
            basic_setting.filter_cases = 3

        print(get_full_filename(dataset,m, basic_setting))
        if result_exists(dataset, m, basic_setting):
            continue

        d.prepare(basic_setting)

        r = m.test(m.train(d.train), d.test_orig)

        save_results(r, d.name, m.name, basic_setting)


def test_standard(dataset, m):
    d = get_data(dataset)

    print(get_full_filename(dataset, m, setting.STANDARD))
    if result_exists(dataset, m, setting.STANDARD):
        return

    d.prepare(setting.STANDARD)

    r = m.test(m.train(d.train), d.test_orig)

    save_results(r, d.name, m.name, setting.STANDARD)


def test_base_comparison():
    tests = [("LIN", setting.LIN), ("TAX", setting.TAX), ("CAMARGO", setting.CAMARGO),
             ("PASQUADIBISCEGLIE", setting.PASQUADIBISCEGLIE), ("SDL", setting.STANDARD), ("DBN", setting.CAMARGO),
             ("TAYMOURI", setting.TAYMOURI)]

    for test in tests:
        print("Test", test[0])
        d = get_data("Helpdesk")
        m = Methods.get_prediction_method(test[0])
        s = test[1]
        if test[0] == "LIN":
            s.filter_cases = 3

        if result_exists("Helpdesk", m, s):
            continue

        d.prepare(s)

        r = m.test(m.train(d.train), d.test_orig)

        save_results(r, d.name, m.name, s)

    # Di Mauro k-fold test
    d = get_data("Helpdesk")
    m = Methods.get_prediction_method("DIMAURO")
    s = setting.DIMAURO

    if result_exists("Helpdesk", m, s):
        return

    d.prepare(s)

    r = m.k_fold_validation(d)
    save_results(r, d.name, m.name, s)


def test_stability():
    results = {}
    d = get_data("Helpdesk")
    d.prepare(setting.STANDARD)
    for method_name in ["SDL", "CAMARGO", "DIMAURO", "LIN", "PASQUADIBISCEGLIE", "TAX", "TAYMOURI"]:
        results[method_name] = []
        m = Methods.get_prediction_method(method_name)
        for _ in range(10):
            r = m.test(m.train(d.train), d.test_orig)
            results[method_name].append(ACCURACY.calculate(r))

    for m in results:
        print(m, "\t".join([str(a) for a in results[m]]))


def ranking_experiments():
    for d in ["BPIC11"]:
        for s in ALL_SETTINGS:
            event_data = get_data(d)
            event_data.prepare(s)

            for method in ALL_METHODS:
                try:
                    m = Methods.get_prediction_method(method)
                    if result_exists(d, m, s):
                        continue
                    if s.train_split == "k-fold":
                        r = m.k_fold_validation(event_data)
                    else:
                        r = m.test(m.train(event_data.train), event_data.test_orig)
                    save_results(r, d, m.name, s)
                except:
                    traceback.print_exc()


def get_scores(d, m, s):
    path, filename = get_filename(d, m, s)
    if os.path.isfile("/".join(path) + "/" + filename):
        results = []
        with open("/".join(path) + "/" + filename) as finn:
            for line in finn:
                results.append(tuple(line.replace("\n", "").split(",")))
        return results


def check_result_files():
    for d in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        print("DATASET: %s" % d)
        for m in ALL_METHODS:
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


if __name__ == "__main__":
    import Methods

    test_base_comparison()
    check_result_files()
    ranking_experiments()
    test_stability()

    for d in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        for m in ["SDL", "CAMARGO", "DBN", "DIMAURO", "LIN", "PASQUADIBISCEGLIE", "TAX", "TAYMOURI"]:
            try:
                test_standard(d, Methods.get_prediction_method(m))
                test_k(d, Methods.get_prediction_method(m))
                test_split(d, Methods.get_prediction_method(m))
                test_filter(d, Methods.get_prediction_method(m))
                test_percentage(d, Methods.get_prediction_method(m))
                test_split_cases(d, Methods.get_prediction_method(m))
                test_end_event(d, Methods.get_prediction_method(m))
            except:
                traceback.print_exc()


