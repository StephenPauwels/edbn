import traceback
from copy import copy

from data import all_data, get_data
from method import ALL as ALL_METHODS
from setting import ALL as ALL_SETTINGS
from metric import ACCURACY
import time
import os

import setting

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
        d = data.get_data(dataset)
        basic_setting.train_split = split
        d.prepare(basic_setting)

        print(get_full_filename(dataset, m, basic_setting))
        if result_exists(dataset, m, basic_setting):
            continue

        if split == "k-fold":
            r = m.k_fold_validation(d)
        else:
            r = m.test(m.train(d.train), d.test_orig)

        save_results(r, d.name, m.name, basic_setting)


def test_k(dataset, m):
    ks = [3, 5, 10]
    basic_setting = copy(setting.STANDARD)
    basic_setting.train_split = "k-fold"

    for k in ks:
        d = data.get_data(dataset)
        basic_setting.train_k = k
        d.prepare(basic_setting)

        print(get_full_filename(dataset, m, basic_setting))
        if result_exists(dataset, m, basic_setting):
            continue

        r = m.k_fold_validation(d)

        save_results(r, d.name, m.name, basic_setting)


def test_percentage(dataset, m):
    train_percentages = [60, 66, 70, 80]
    basic_setting = copy(setting.STANDARD)

    for percentage in train_percentages:
        d = data.get_data(dataset)
        basic_setting.train_percentage = percentage
        d.prepare(basic_setting)

        print(get_full_filename(dataset,m, basic_setting))
        if result_exists(dataset, m, basic_setting):
            continue

        r = m.test(m.train(d.train), d.test_orig)

        save_results(r, d.name, m.name, basic_setting)


def test_end_event(dataset, m):
    end_events = [True, False]
    basic_setting = copy(setting.STANDARD)

    for end_event in end_events:
        d = data.get_data(dataset)
        basic_setting.add_end = end_event
        d.prepare(basic_setting)

        print(get_full_filename(dataset,m, basic_setting))
        if result_exists(dataset, m, basic_setting):
            continue

        r = m.test(m.train(d.train), d.test_orig)

        save_results(r, d.name, m.name, basic_setting)


def test_split_cases(dataset, m):
    split_cases = [True, False]
    basic_setting = copy(setting.STANDARD)

    for split_case in split_cases:
        d = data.get_data(dataset)
        basic_setting.split_cases = split_case
        d.prepare(basic_setting)

        print(get_full_filename(dataset,m, basic_setting))
        if result_exists(dataset, m, basic_setting):
            continue

        r = m.test(m.train(d.train), d.test_orig)

        save_results(r, d.name, m.name, basic_setting)


def test_filter(dataset, m):
    filters = [None, 5]
    basic_setting = copy(setting.STANDARD)

    for filter in filters:
        d = data.get_data(dataset)
        basic_setting.filter_cases = filter
        if filter == 5 and dataset == "Helpdesk":
            basic_setting.filter_cases = 3
        d.prepare(basic_setting)

        print(get_full_filename(dataset,m, basic_setting))
        if result_exists(dataset, m, basic_setting):
            continue

        r = m.test(m.train(d.train), d.test_orig)

        save_results(r, d.name, m.name, basic_setting)


def test_standard(dataset, m):
    d = data.get_data(dataset)
    d.prepare(setting.STANDARD)

    print(get_full_filename(dataset, m, setting.STANDARD))
    if result_exists(dataset, m, setting.STANDARD):
        return

    r = m.test(m.train(d.train), d.test_orig)

    save_results(r, d.name, m.name, setting.STANDARD)


def test_base_comparison():
    tests = [("LIN", setting.LIN), ("TAX", setting.TAX), ("CAMARGO", setting.CAMARGO), ("DIMAURO", setting.DIMAURO),
             ("PASQUADIBISCEGLIE", setting.PASQUADIBISCEGLIE), ("SDL", setting.STANDARD), ("DBN", setting.CAMARGO)]

    for test in tests:
        d = data.get_data("Helpdesk")
        m = method.get_method(test[0])
        s = test[1]

        d.prepare(s)
        if result_exists("Helpdesk", m, s):
            continue

        r = m.test(m.train(d.train), d.test_orig)

        save_results(r, d.name, m.name, s)

def ranking_experiments(output_file):
    for d in all_data:
        event_data = get_data(d)
        for s in ALL_SETTINGS:
            event_data.prepare(s)

            with open(output_file, "a") as fout:
                fout.write("Data: " + event_data.name + "\n")
                fout.write(s.to_file_str())
                fout.write("Date: " + time.strftime("%d.%m.%y-%H.%M", time.localtime()) + "\n")
                fout.write("------------------------------------\n")

            for m in ALL_METHODS:
                m.train(event_data)
                acc = m.test(event_data, ACCURACY)
                with open(output_file, "a") as fout:
                    fout.write(m.name + ": " + str(acc) + "\n")
            with open(output_file, "a") as fout:
                fout.write("====================================\n\n")


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
        brier = metric.BRIER.calculate(results)
        print("Brier:", brier)
        prec = metric.PRECISION.calculate(results)
        print("Precision:", prec)
        recall = metric.RECALL.calculate(results)
        print("Recall:", recall)


def check_result_files():
    import method

    for d in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        print("DATASET: %s" % d)
        for m in method.ALL:
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
    import method
    import data

    test_base_comparison()
    # check_result_files()
    # ranking_experiments("ranking_results.txt")

    #
    # for d in ["BPIC11"]: #["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
    #     for m in ["TAYMOURI"]:
    #         try:
    #             print("TEST Standard")
    #             test_standard(d, method.get_method(m))
    #             print("TEST k")
    #             test_k(d, method.get_method(m))
    #             test_split(d, method.get_method(m))
    #             test_filter(d, method.get_method(m))
    #             test_percentage(d, method.get_method(m))
    #             test_split_cases(d, method.get_method(m))
    #             test_end_event(d, method.get_method(m))
    #         except:
    #             traceback.print_exc()

    # s = setting.STANDARD
    # s.train_split = "k-fold"
    # s.train_k = 10
    # get_scores("Helpdesk", "DBN", s)
    # get_scores("Helpdesk", "LIN", s)
    # get_scores("Helpdesk", "DIMAURO", s)
    # get_scores("Helpdesk", "CAMARGO", s)
    # get_scores("Helpdesk", "PASQUADIBISCEGLIE", s)
    # get_scores("Helpdesk", "TAX", s)
    # get_scores("Helpdesk", "SDL", s)
    # get_scores("Helpdesk", "TAYMOURI", s)

