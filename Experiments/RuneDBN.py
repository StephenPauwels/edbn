#import eDBN.Execute as edbn
import Utils.Utils as utils

import pandas as pd
import eDBN.Execute as edbn
from Utils.LogFile import LogFile
from Utils.BPIPreProcess import preProcessData
import Utils.PlotResults as plot

if __name__ == "__main__":
    # Use the BPIC15_x_sorted.csv to generate new training and test datafiles with anomalies introduced
    # After running this once you can comment this line out
    preProcessData("../Data/")

    # Indicate which are the training and test files
    train_file = "../Data/BPIC15_train_1.csv"
    test_file = "../Data/BPIC15_test_1.csv"

    # Load logfile to use as training data
    train_data = LogFile(train_file, ",", header=0, rows=100000, time_attr=None, trace_attr="Case")
    train_data.remove_attributes(["Anomaly"])

    # Train the model
    model = edbn.train(train_data)

    # Test the model and save the scores in ../Data/output.csv
    test_data = LogFile(test_file, ",", header=0, rows=100000, time_attr=None, trace_attr="Case",
                        string_2_int=train_data.string_2_int, int_2_string=train_data.int_2_string)
    edbn.test(test_data, "../Data/output.csv", model, label = "Anomaly", normal_val = "0")

    # Plot the ROC curve based on the results
    plot.plot_single_roc_curve("../Data/output.csv")