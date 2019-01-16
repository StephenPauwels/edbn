#import eDBN.Execute as edbn
import Utils.Utils as utils

import pandas as pd
import eDBN.Execute as edbn
from Utils.LogFile import LogFile
from Utils.BPIPreProcess import preProcessData
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

    # Indicate which are the training and test files
    train_file = "../Data/bpic15_1_train.csv"
    test_file = "../Data/bpic15_1_test.csv"

    # Load logfile to use as training data
    train_data = LogFile(train_file, ",", 0, 500000, None, "Case_ID", activity_attr="Activity")
    train_data.remove_attributes(["Anomaly"])
    #train_data.keep_attributes(["Case_ID", "Complete_Timestamp", "Activity", "Resource", "case_termName"])
    train_data.remove_attributes(["planned", "dueDate", "dateStop", "dateFinished"])
    train_data.remove_attributes(["case_IDofConceptCase"])
    #train_data.remove_attributes(["case_Includes_subCases"])
    #train_data.remove_attributes(["case_Responsible_actor"])
    train_data.remove_attributes(["case_SUMleges"])
    #train_data.remove_attributes(["case_caseProcedure"])
    #train_data.remove_attributes(["case_caseStatus"])
    #train_data.remove_attributes(["case_case_type"])
    train_data.remove_attributes(["case_landRegisterID"])
    #train_data.remove_attributes(["case_last_phase"])
    train_data.remove_attributes(["case_parts"])
    #train_data.remove_attributes(["case_requestComplete"])
    #train_data.remove_attributes(["case_termName"])

    #train_data.remove_attributes(["lifecycle_transition"])

    # Train the model
    model = edbn.train(train_data)

    # Test the model and save the scores in ../Data/output.csv
    test_data = LogFile(test_file, ",", header=0, rows=500000, time_attr=None, trace_attr="Case_ID",
                        values=train_data.values)
    edbn.test(test_data, "../Data/output.csv", model, label = "Anomaly", normal_val = "0")

    # Plot the ROC curve based on the results
    plot.plot_single_roc_curve("../Data/output.csv")
    plot.plot_single_prec_recall_curve("../Data/output.csv")

if __name__ == "__main__":
    #run_reduced()
    run_full()