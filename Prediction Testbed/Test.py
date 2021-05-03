import method
from bise_experiments import check_result_files, get_scores
import data
import setting
import metric

from copy import copy


def process_standard():
    for dataset in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        print("DATASET:", dataset)
        for m in ["SDL", "DBN", "CAMARGO", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "TAX", "TAYMOURI"]:
            results = get_scores(dataset, m, setting.STANDARD)
            if results:
                acc = metric.ACCURACY.calculate(results)
            else:
                acc = 0
            print(m, acc, sep="\t")
        print()

def process_k():
    ks = [3, 5, 10]
    basic_setting = copy(setting.STANDARD)
    basic_setting.train_split = "k-fold"

    for dataset in ["Helpdesk", "BPIC12W", "BPIC12", "BPIC11", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        print()
        for m in ["SDL", "DBN", "CAMARGO", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "TAX", "TAYMOURI"]:
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
        for m in ["SDL", "DBN", "CAMARGO", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "TAX", "TAYMOURI"]:
            accs = []
            for split in splits:
                basic_setting.train_split = split
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
        for m in ["SDL", "DBN", "CAMARGO", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "TAX", "TAYMOURI"]:
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
        for m in ["SDL", "DBN", "CAMARGO", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "TAX", "TAYMOURI"]:
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
        for m in ["SDL", "DBN", "CAMARGO", "Di Mauro", "LIN", "PASQUADIBISCEGLIE", "TAX", "TAYMOURI"]:
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


if __name__ == "__main__":
    RESULT_FOLDER = "bise_results"

    check_result_files()
    # process_standard()
    # process_k()
    # process_split()
    # process_percentage()
    # process_end_event()
    # process_split_cases()