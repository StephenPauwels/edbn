"""
    File used for the Comparison Experiments in the SAC'19 paper

    Author: Stephen Pauwels
"""

import RelatedMethods.Bohmer.Execute as bmr
import EDBN.Execute as edbn
import Utils.BPIPreProcess as preprocess
import Utils.PlotResults as plt
from Utils.LogFile import LogFile


def compare_bpics(path):
    for i in range(1,6):
        # Input Files
        train = path + "BPIC15_train_%i.csv" % (i)
        test = path + "BPIC15_test_%i.csv" % (i)
        output = path + "Output/BPIC15_output_%i.csv" % (i)
        output_edbn = path + "Output/BPIC15_edbn_output_%i.csv" % (i)
        prec_recall = path + "Output/prec_recall_%i.png" % (i)
        roc = path + "Output/roc_%i.png" % (i)

        train_data = LogFile(train, ",", 0, 500000, "Time", "Case", activity_attr="Activity", convert=False)
        train_data.remove_attributes(["Anomaly", "Type", "Time"])
        test_data = LogFile(test, ",", 0, 500000, "Time", "Case", activity_attr="Activity", values=train_data.values, convert=False)

        bohmer_model = bmr.train(train_data, 0, 1, 2)
        bmr.test(test_data, output, bohmer_model, label = "Anomaly", normal_val = "0")

        train_data.convert2int()
        test_data.convert2int()

        edbn_model = edbn.train(train_data, only_activity=False)
        edbn.test(test_data, output_edbn, edbn_model, label = "Anomaly", normal_val = "0")

        plt.plot_compare_prec_recall_curve([output, output_edbn], ["Likelihood Graph", "EDBN"], save_file=prec_recall)
        plt.plot_compare_roc_curve([output, output_edbn], ["Likelihood Graph", "EDBN"], save_file=roc)

def compare_bpic_total(path):
    train = path + "BPIC15_train_total.csv"
    test = path + "BPIC15_test_total.csv"
    output = path + "Output/BPIC_15_output_total.csv"
    output_edbn = path + "Output/BPIC15_edbn_output_total.csv"
    prec_recall = path + "Output/prec_recall_total.png"
    roc = path + "Output/roc_total.png"

    train_data = LogFile(train, ",", 0, 500000, "Time", "Case", activity_attr="Activity", convert=False)
    train_data.remove_attributes(["Anomaly", "Type", "Time"])
    test_data = LogFile(test, ",", 0, 500000, "Time", "Case", activity_attr="Activity", values=train_data.values, convert=False)

    bohmer_model = bmr.train(train_data)
    bmr.test(test_data, output, bohmer_model, label = "Anomaly", normal_val = "0")

    train_data.convert2int()
    test_data.convert2int()

    edbn_model = edbn.train(train_data, only_activity=False)
    edbn.test(test_data, output_edbn, edbn_model, label = "Anomaly", normal_val = "0")

    plt.plot_compare_prec_recall_curve([output, output_edbn], ["Likelihood Graph", "EDBN"], save_file=prec_recall)
    plt.plot_compare_roc_curve([output, output_edbn], ["Likelihood Graph", "EDBN"], roc)

if __name__  == "__main__":
    path = "../Data/"

    # preprocess.preProcessData(path)
    # preprocess.preProcessData_total(path)

    compare_bpics(path)
    compare_bpic_total(path)