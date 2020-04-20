import time

from Utils.LogFile import LogFile

from RelatedMethods.Tax import adapter as tax
from RelatedMethods.Taymouri import adapter as taymouri
from RelatedMethods.Camargo import adapter as camargo
from RelatedMethods.Lin import adapter as lin
from RelatedMethods.DiMauro import adapter as dimauro
from Predictions import edbn_adapter as edbn
from Predictions import base_adapter as baseline

def run_experiment(data, prefix_size, add_end_event, split_method, split_cases, train_percentage):
    logfile = LogFile(data, ",", 0, None, None, "case",
                      activity_attr="event", convert=False, k=prefix_size)
    if add_end_event:
        logfile.add_end_events()
    logfile.keep_attributes(["case", "event", "role"])
    logfile.convert2int()
    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(train_percentage, case=split_cases, method=split_method)

    with open("results.txt", "a") as fout:
        fout.write("Data: " + data)
        fout.write("\nPrefix Size: " + str(prefix_size))
        fout.write("\nEnd event: " + str(add_end_event))
        fout.write("\nSplit method: " + split_method)
        fout.write("\nSplit cases: " + str(split_cases))
        fout.write("\nTrain percentage: " + str(train_percentage))
        fout.write("\nDate: " + time.strftime("%d.%m.%y-%H.%M", time.localtime()))
        fout.write("------------------------------------")
        tax_acc = tax.test(test_log, tax.train(train_log, epochs=100, early_stop=10))
        fout.write("\nTax: " + str(tax_acc))
        fout.write("\n")

        train_data, test_data = taymouri.create_input(train_log, test_log, 5)
        taymouri_acc = taymouri.test(test_data, taymouri.train(train_data))
        fout.write("Taymouri: " + str(taymouri_acc))
        fout.write("\n")

        camargo_acc = camargo.test(test_log, camargo.train(train_log, epochs=100, early_stop=10))
        fout.write("Camargo: " + str(camargo_acc))
        fout.write("\n")

        lin_acc = lin.test(test_log, lin.train(train_log, epochs=100, early_stop=10))
        fout.write("Lin: " + str(lin_acc))
        fout.write("\n")

        dimauro_acc = dimauro.test(test_log, dimauro.train(train_log, epochs=100, early_stop=10))
        fout.write("Di Mauro: " + str(dimauro_acc))
        fout.write("\n")

        edbn_acc = edbn.test(test_log, edbn.train(train_log))
        fout.write("EDBN: " + str(edbn_acc))
        fout.write("\n")

        baseline_acc = baseline.test(test_log, baseline.train(train_log, epochs=100, early_stop=10))
        fout.write("Baseline: " + str(baseline_acc))
        fout.write("\n")
        fout.write("====================================\n\n")

    print("ACCURACIES")
    print("Tax:", tax_acc)
    print("Taymouri:", taymouri_acc)
    print("Camargo:", camargo_acc)
    print("Lin:", lin_acc)
    print("Di Mauro:", dimauro_acc)
    print("EDBN:", edbn_acc)
    print("Baseline:", baseline_acc)

def experiments_helpdesk():
    data = "../Data/Camargo_Helpdesk.csv"
    prefix_size = [1,2,3,4,5,6]
    add_end_event = [True, False]
    split_method = ["train-test", "test-train", "random"]
    split_cases = [True, False]
    train_percentage = [70, 80]

    for ps in prefix_size:
        for aee in add_end_event:
            for sm in split_method:
                for sc in split_cases:
                    for tp in train_percentage:
                        run_experiment(data, ps, aee, sm, sc, tp)


if __name__ == "__main__":
    data = ["../Data/BPIC15_1_sorted_new.csv"
    prefix_size = 5
    add_end_event = False
    split_method = "train-test"
    split_cases = True
    train_percentage = 70


