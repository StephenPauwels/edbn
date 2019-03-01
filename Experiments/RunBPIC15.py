#import eDBN.Execute as edbn
import Utils.Utils as utils

import pandas as pd
import eDBN.Execute as edbn
from Utils.LogFile import LogFile
from Utils.BPIPreProcess_General import preProcessData
import Utils.PlotResults as plot

def run_reduced():
    # Use the BPIC15_x_sorted.csv to generate new training and test datafiles with anomalies introduced
    # After running this once you can comment this line out
    #preProcessData("../Data/")

    # Indicate which are the training and test files
    train_file = "../Data/BPIC15_train_1.csv"
    test_file = "../Data/BPIC15_test_1.csv"

    # Load logfile to use as training data
    train_data = LogFile(train_file, ",", 0, 500000, "Time", "Case", activity_attr="Activity")
    train_data.remove_attributes(["Anomaly", "Type"])

    # Train the model
    model = edbn.train(train_data)

    # Test the model and save the scores in ../Data/output.csv
    test_data = LogFile(test_file, ",", header=0, rows=500000, time_attr="Time", trace_attr="Case",
                        values=train_data.values)
    edbn.test(test_data, "../Data/output.csv", model, label = "Anomaly", normal_val = "0")

    # Plot the ROC curve based on the results
    plot.plot_single_roc_curve("../Data/output.csv")
    plot.plot_single_prec_recall_curve("../Data/output.csv")

def run_full():
    # Use the BPIC15_x_sorted.csv to generate new training and test datafiles with anomalies introduced
    # After running this once you can comment this line out
    #preProcessData("../Data/")


    for i in range(1,2):

        # Indicate which are the training and test files
        train_file = "../Data/bpic15_%i_train.csv" % (i)
        test_file = "../Data/bpic15_%i_test.csv" % (i)

        # Load logfile to use as training data
        train_data = LogFile(train_file, ",", 0, 500000, time_attr="Complete_Timestamp", trace_attr="Case_ID", activity_attr="Activity")
        train_data.remove_attributes(["Anomaly"])
    #    train_data.keep_attributes(["Case_ID", "Complete_Timestamp", "Activity", "Resource", "case_termName"])
        train_data.remove_attributes(["planned"])
        train_data.remove_attributes(["dueDate"])
        train_data.remove_attributes(["dateFinished"])

        train_data.keep_attributes(["Case_ID", "Complete_Timestamp", "Activity", "Resource", "Weekday"])

        train_data.create_k_context()
        train_data.add_duration_to_k_context()

        print(train_data.contextdata)

        # Train the model
        model = edbn.train(train_data)

        # Test the model and save the scores in ../Data/output.csv
        test_data = LogFile(test_file, ",", header=0, rows=500000, time_attr="Complete_Timestamp", trace_attr="Case_ID",
                            values=train_data.values)
        test_data.create_k_context()
        test_data.add_duration_to_k_context()

        edbn.test(test_data, "../Data/output2_%i.csv" % (i), model, label = "Anomaly", normal_val = "0", train_data=train_data)

        # Plot the ROC curve based on the results
        plot.plot_single_roc_curve("../Data/output2_%i.csv" % (i), title="BPIC15_%i" % (i))
        plot.plot_single_prec_recall_curve("../Data/output2_%i.csv" % (i), title="BPIC15_%i" % (i))

    out_files = []
    labels = []
    for i in range(1,6):
        out_files.append("../Data/output2_%i.csv" % (i))
        labels.append("MUNIS_%i" % (i))
    plot.plot_compare_roc_curve(out_files, labels, "BPIC15 Comparison")
    plot.plot_compare_prec_recall_curve(out_files, labels, "BPIC15 Comparison")

if __name__ == "__main__":
    #run_reduced()
    run_full()