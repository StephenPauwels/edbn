#matplotlib.use("Agg")  # Disable comment when no output should be shown (for running on servers without graphical interface)

import datetime
import multiprocessing as mp
import random

import pandas as pd
from joblib import Parallel, delayed

import Bohmer.LikelihoodGraph as lg

def preprocess(data, case_attr, activity_attr, time_attr, time_anoms):
    """
    Function to split original data in training and test data with the introduction of anomalies.
    Anomalies are generated according to the explanation in Bohmer paper

    """

    training_traces = []
    test_traces = []

    def convert_2_weekdays(row):
        row["Weekday"] = datetime.datetime.strptime(row[time_attr], "%Y-%m-%d %H:%M:%S").weekday()
        row[time_attr] = datetime.datetime.strptime(row[time_attr], "%Y-%m-%d %H:%M:%S")
        return row

    data = data.apply(func=convert_2_weekdays, axis=1)

    traces = data.groupby([case_attr])
    for trace in traces:
        trace_id = trace[0]
        trace_data = trace[1].copy()
        if random.randint(0,1) == 0: # Adding to Train Log
            trace_data["Anomaly"] = "0"
            training_traces.append(trace_data)
        else:
            if random.randint(1,100) > 50: # No anomaly injection with 50% chance
                trace_data["Anomaly"] = "0"
                test_traces.append(trace_data)
            else:
                anom_trace, types = introduce_anomaly(trace_data, [case_attr, time_attr], ["Resource", "Activity", "Weekday"], single=False, time_attr=time_attr, time_anoms=time_anoms)
                anom_trace["anom_types"] = str(types)
                test_traces.append(anom_trace)

    return pd.concat(training_traces, sort=False), pd.concat(test_traces, sort=False)


def introduce_anomaly(trace, ignore_attrs = None, anom_attributes = None, single = False, time_attr = None, time_anoms = True):
    """
    Add anomaly to the input trace

    :param trace: input trace, containing no anomalies
    :return: trace containing anomalies
    """
    def alter_activity_order(trace):
        """
        Switch two consecutive events
        """
        if len(trace) == 1:
            return trace
        alter = random.randint(0, len(trace) - 2)
        alter_index = trace.index[alter]
        alter_index2 = trace.index[alter + 1]
        tmp = trace.loc[alter_index].copy()
        tmp_time1 = trace.loc[alter_index][time_attr]
        tmp_time2 = trace.loc[alter_index2][time_attr]
        trace.loc[alter_index] = trace.loc[alter_index2]
        trace.loc[alter_index2] = tmp
        trace.loc[alter_index, time_attr] = tmp_time1
        trace.loc[alter_index2, time_attr] = tmp_time2
        return trace


    def new_value(trace, attribute):
        """
        Add generic new value to attribute
        """
        trace.loc[random.choice(trace.index), attribute] = "NEW_%s" % (attribute)
        return trace


    def duration_anomaly(trace, time_attr):
        """
        Introduce a timing anomaly and update next times
        """
        used_events = []

        for _ in range(random.randint(1,1)): # For now, only add one time anomaly
            start_anom = random.randint(2, len(trace) - 1)
            while start_anom in used_events and len(used_events) < len(trace) - 2:
                start_anom = random.randint(2, len(trace) - 1)
            used_events.append(start_anom)
            delta = datetime.timedelta(1 + int(abs(random.gauss(1000,100))))

            for i in range(start_anom, len(trace)):
                trace.loc[trace.index[i], time_attr] = trace.loc[trace.index[i], time_attr] + delta
                if i == start_anom:
                    trace.loc[trace.index[i], "time_anomaly"] = "Changed"
        trace["Anomaly"] = 1
        return True,trace

    def generate_Anomaly(trace, num_diff_anoms, from_nums, to_nums):
        """
        Introduce anomaly in the trace according to the input parameters
        """
        anoms = set()
        anomaly_types = []
        for _ in range(num_diff_anoms):
            if time_anoms and len(trace) > 3 and random.randint(1,10) < 3:
                anom_added, trace = duration_anomaly(trace,time_attr)
                if anom_added:
                    anomaly_types.append("alter_duration")
            else:
                anomaly = random.randint(0,len(anom_attributes))
                while anomaly in anoms: # Ensure each type of anomaly is only choosen once
                    anomaly = random.randint(0, len(anom_attributes))
                anoms.add(anomaly)

                for j in range(random.randint(from_nums, to_nums)):
                    if anomaly == len(anom_attributes):
                        trace = alter_activity_order(trace)
                        anomaly_types.append("alter_order")
                    else:
                        trace = new_value(trace, anom_attributes[anomaly])
                        anomaly_types.append("new_%s" % (anom_attributes[anomaly]))
                trace["Anomaly"] = 1
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



def generate(file, train_file, test_file, case_attr, activity_attr, time_attr, time_anom):
    data = pd.read_csv(file, header=0, delimiter=",", dtype="str")
    new_cols = []
    for col in list(data.columns):
        new_cols.append(col.replace(":", "_").replace(" ", "_").replace("(", "").replace(")", ""))
    data.columns = new_cols

    training_data, testing_data = preprocess(data, case_attr, activity_attr, time_attr, time_anom)

    training_data.to_csv(train_file, index=False)
    testing_data.to_csv(test_file, index=False)

def generate_combined(files, train_file, test_file, case_attr, activity_attr, time_attr, time_anom):
    logs = []
    for file in files:
        data = pd.read_csv(file, header=0, delimiter=",", dtype="str")
        new_cols = []
        for col in list(data.columns):
            new_cols.append(col.replace(":", "_").replace(" ", "_").replace("(", "").replace(")", ""))
        data.columns = new_cols
        logs.append(data)
    data = pd.concat(logs)

    training_data, testing_data = preprocess(data, case_attr, activity_attr, time_attr, time_anom)

    training_data.to_csv(train_file, index=False)
    testing_data.to_csv(test_file, index=False)


def sort_datafile(file, outfile, time_attr):
    df = pd.read_csv(file, header=0, delimiter=",", dtype="str")
    df = df.apply(func=convert_dates, axis=1)
    df = df.sort_values(by=[time_attr, "Activity"], kind="mergesort")
    df.to_csv(outfile, index=False)

def convert_dates(row):
    time_attr= "Complete Timestamp"
    row[time_attr] = datetime.datetime.strptime(row[time_attr], "%m/%d/%y %H:%M:%S")
    return row


def generate_discrete_bpic15(path = "../Data/"):
    """
    Create training and test file from BPIC15 data, only introducing discrete anomalies
    """
    files = []
    train = []
    test = []
    for i in range(1,6):
        files.append("../Data/BPIC15_%i_sorted_new.csv" % (i))
        train.append("../Data/bpic15_%i_train.csv" % (i))
        test.append("../Data/bpic15_%i_test.csv" % (i))

    for i in range(len(files)):
        print("PREPROCESS: Creating", files[i])
        generate(files[i], train[i], test[i], "Case_ID", "Activity", "Complete_Timestamp", False)

def generate_bpic15(path = "../Data/"):
    """
    Create training and test file from BPIC15 data, only introducing discrete anomalies
    """
    files = []
    train = []
    test = []
    for i in range(1,2):
        files.append("../Data/BPIC15_%i_sorted_new.csv" % (i))
        train.append("../Data/bpic15_%i_train.csv" % (i))
        test.append("../Data/bpic15_%i_test.csv" % (i))

    for i in range(len(files)):
        print("PREPROCESS: Creating", files[i])
        generate(files[i], train[i], test[i], "Case_ID", "Activity", "Complete_Timestamp", True)
#    generate_combined(files, "../Data/bpic15_total_train.csv", "../Data/bpic15_total_train.csv", "Case_ID", "Activity", "Complete_Timestamp", True)

if __name__ == "__main__":
    #generate_discrete_bpic15()
    generate_bpic15()
