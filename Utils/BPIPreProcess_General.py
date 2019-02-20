#matplotlib.use("Agg")  # Disable comment when no output should be shown (for running on servers without graphical interface)

import datetime
import multiprocessing as mp
import random

import pandas as pd
from joblib import Parallel, delayed

import Bohmer.LikelihoodGraph as lg



def preprocess(data, case_attr, activity_attr, time_attr):
    """
    Function to split original data in training and test data with the introduction of anomalies.
    Anomalies are generated according to the explanation in Bohmer paper

    """

    training_traces = []
    test_traces = []

    def convert_2_weekdays(row):
        row["Weekday"] = datetime.datetime.strptime(row[time_attr], "%m/%d/%y %H:%M:%S").weekday()
        row[time_attr] = datetime.datetime.strptime(row[time_attr], "%m/%d/%y %H:%M:%S")
        return row

    data = data.apply(func=convert_2_weekdays, axis=1)

    traces = data.groupby([case_attr])
    for trace in traces:
        trace_id = trace[0]
        trace_data = trace[1].copy()
        if random.randint(0,1) == 0: # Adding to Train Log
            trace_data["Anomaly"] = "0"
            training_traces.append(trace_data)
        else: # Adding to Test Log
            if random.randint(0,100) > 50: # No anomaly injection with 50% chance
                trace_data["Anomaly"] = "0"
                test_traces.append(trace_data)
            else:
                anom_trace, types = introduce_anomaly(trace_data, [case_attr, time_attr], ["Resource", "Activity", "Weekday"], single=False, time_attr=time_attr)
                anom_trace["Anomaly"] = "1"
                anom_trace["anom_types"] = str(types)
                test_traces.append(anom_trace)

    return pd.concat(training_traces, sort=False), pd.concat(test_traces, sort=False)


def introduce_anomaly(trace, ignore_attrs = None, anom_attributes = None, single = False, time_attr = None):
    """
    Add anomaly to the input trace

    :param trace: input trace, containing no anomalies
    :return: trace containing anomalies
    """
    def alter_activity_order(trace):
        if len(trace) == 1:
            return trace
        alter = random.randint(0, len(trace) - 2)
        tmp = trace.iloc[alter].copy()
        trace.iloc[alter] = trace.iloc[alter + 1]
        trace.iloc[alter + 1] = tmp
        return trace


    def new_value(trace, attribute):
        trace.loc[random.choice(trace.index), attribute] = "NEW_%s" % (attribute)
        return trace


    def duration_anomaly(trace, time_attr):
        if len(trace) < 4:
            return trace
        start_anom = random.randint(2, len(trace) - 1)

        delta = datetime.timedelta(1 + abs(random.gauss(4,2)))

        for i in range(start_anom, len(trace)):
            trace.loc[trace.index[i], time_attr] = trace.loc[trace.index[i], time_attr] + delta

        return trace

    def generate_Anomaly(trace, num_diff_anoms, from_nums, to_nums):
        anoms = set()
        anomaly_types = []
        for i in range(num_diff_anoms):

            anomaly = len(anom_attributes) + 4 #random.randint(0,len(anom_attributes))
            while anomaly in anoms: # Ensure each type of anomaly is only choosen once
                anomaly = len(anom_attributes) + 4 #random.randint(0, len(anom_attributes))

            if anomaly > len(anom_attributes):
                for j in range(random.randint(from_nums, to_nums)):
                    trace = duration_anomaly(trace, time_attr)
                    anomaly_types.append("alter_duration")
                break
            else:
                for j in range(random.randint(from_nums, to_nums)):
                    if anomaly == len(anom_attributes):
                        trace = alter_activity_order(trace)
                        anomaly_types.append("alter_order")
                    else:
                        trace = new_value(trace, anom_attributes[anomaly])
                        anomaly_types.append("new_%s" % (anom_attributes[anomaly]))

        return (trace, anomaly_types)

    if anom_attributes is None:
        anom_attributes = list(trace.columns)
        if ignore_attrs is not None:
            for a in ignore_attrs:
                anom_attributes.remove(a)

    density = random.randint(1,3)
    if density == 1:
        trace = generate_Anomaly(trace, 1, 2, 4)
    elif density == 2:
        trace = generate_Anomaly(trace, 2, 1, 3)
    elif density == 3:
        if random.randint(0,1) == 0:
            trace = generate_Anomaly(trace, 3, 1, 2)
        else:
            trace = generate_Anomaly(trace, 4, 1, 2)

    return trace



def preProcessData(file, train_file, test_file, case_attr, activity_attr, time_attr):
    data = pd.read_csv(file, header=0, delimiter=",", dtype="str")
    new_cols = []
    for col in list(data.columns):
        new_cols.append(col.replace(":", "_").replace(" ", "_").replace("(", "").replace(")", ""))
    data.columns = new_cols
    training_data, testing_data = preprocess(data, case_attr, activity_attr, time_attr)
    training_data = training_data.sort_values(by=[time_attr])
    training_data.to_csv(train_file, index=False)
    testing_data = testing_data.sort_values(by=[time_attr])
    testing_data.to_csv(test_file, index=False)

def preProcessData_total(files, train_file, test_file, case_attr, activity_attr, time_attr):
    logs = []
    for file in files:
        data = pd.read_csv(file, header=0, delimiter=",", dtype="str")
        new_cols = []
        for col in list(data.columns):
            new_cols.append(col.replace(":", "_").replace(" ", "_").replace("(", "").replace(")", ""))
        data.columns = new_cols
        logs.append(data)
    data = pd.concat(logs)

    training_data, testing_data = preprocess(data, case_attr, activity_attr, time_attr)
    training_data.to_csv(train_file, index=False)
    testing_data.to_csv(test_file, index=False)

if __name__ == "__main__":
    files = []
    train_files = []
    test_files = []
    for i in range(1,6):
        files.append("../Data/BPIC15_%i_sorted.csv" % (i))
        train_files.append("../Data/bpic15_%i_train_only_duration.csv" % (i))
        test_files.append("../Data/bpic15_%i_test_only_duration.csv" % (i))

    for i in range(len(files)):
        print("PREPROCESS: Creating", files[i])
        preProcessData(files[i], train_files[i], test_files[i], "Case_ID", "Activity", "Complete_Timestamp")

    print("PREPROCESS: Creating Total")
    #preProcessData_total(files, "../Data/bpic15_total_train_only_duration.csv", "../Data/bpic15_total_test_only_duration.csv", "Case_ID", "Activity", "Complete_Timestamp")
