import Bohmer.Execute as bmr
import eDBN.Execute as edbn
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

        #bohmer_model = bmr.train(train + "_ints", header = 0, length = 500000)
        #bmr.test(train + "_ints", test + "_ints", output, bohmer_model, ",", 500000, skip=0)

        train_data = LogFile(train, ",", 0, 500000, None, "Case")
        train_data.remove_attributes(["Anomaly"])
        test_data = LogFile(test, ",", 0, 500000, None, "Case", train_data.string_2_int, train_data.int_2_string)

        edbn_model = edbn.train(train_data)
        edbn.test(test_data, output_edbn, edbn_model, "Anomaly", "0")

        plt.plot_compare_prec_recall_curve([output, output_edbn], ["Likelihood Graph", "eDBN"], save_file=prec_recall)
        plt.plot_compare_roc_curve([output, output_edbn], ["Likelihood Graph", "eDBN"], roc)

def compare_bpic_total(path):
    train = path + "BPIC15_train_total.csv"
    test = path + "BPIC15_test_total.csv"
    output = path + "Output/BPIC_15_output_total.csv"
    output_edbn = path + "Output/BPIC15_edbn_output_total.csv"
    prec_recall = path + "Output/prec_recall_total.png"
    roc = path + "Output/roc_total.png"

    #bohmer_model = bmr.train(train, header = 0, length = 5000000)
    #bmr.test(train, test, output, bohmer_model, ",", 5000000, skip=0)

    train_data = LogFile(train, ",", 0, 500000, None, "Case")
    train_data.remove_attributes(["Anomaly"])
    test_data = LogFile(test, ",", 0, 500000, None, "Case", train_data.string_2_int, train_data.int_2_string)

    edbn_model = edbn.train(train_data)
    edbn.test(test_data, output_edbn, edbn_model, "Anomaly", "0")

    plt.plot_compare_prec_recall_curve([output, output_edbn], ["Likelihood Graph", "eDBN"], save_file=prec_recall)
    plt.plot_compare_roc_curve([output, output_edbn], ["Likelihood Graph", "eDBN"], roc)

if __name__  == "__main__":
    path = "../Data/"

    #preprocess.preProcessData(path)
    #preprocess.preProcessData_total(path)

   # compare_bpics(path)
    compare_bpic_total(path)