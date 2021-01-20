"""
    Author: Stephen Pauwels
"""


#matplotlib.use("Agg")  # Disable comment when no output should be shown (for running on servers without graphical interface)

import datetime
import random

ACTIVITY_INDEX = 1
RESOURCE_INDEX = 2
WEEKDAY_INDEX = 3

def read_raw_file(file):
    """
    Read original file and group the data by the case number

    :param file: Locatin of original .csv file
    :return: dictionary with data grouped by case
    """
    print("Reading", file)
    output = {}

    with open(file, "r") as fin:
        fin.readline()
        for line in fin:
            line_split = line.split(",")
            date = datetime.datetime.strptime(line_split[3], "%Y-%m-%d %H:%M:%S")
            line = [str(date), "a_" + line_split[1], "r_" + line_split[2], "wd_" + str(date.weekday())]
            if eval(line_split[0]) not in output:
                output[eval(line_split[0])] = []
            output[eval(line_split[0])].append(line)
    return output


def write_to_file(train_file, test_file, log_dict):
    """
    Function to split original data in training and test data with the introduction of anomalies.
    Anomalies are generated according to the explanation in Bohmer paper

    :param train_file: location to save the training file
    :param test_file: location to save the test file
    :param log_dict: dictionary containing the original data grouped by case
    """
    i = 0
    train_events = []
    test_events = []

    for key in log_dict:
        trace = log_dict[key]
        if random.randint(0,1) == 0: # Add file to training set with 50% chance
            for e_idx in range(len(trace)):
                train_events.append(",".join([str(x) for x in trace[e_idx]]) + "," + str(key) + ",0,None")
        else: # Add file to test set
            if random.randint(0,100) > 50: # No anomaly injection with 50% chance
                for e_idx in range(len(trace)):
                    test_events.append(",".join([str(x) for x in trace[e_idx]]) + "," + str(key) + ",0,None")
            else: # Anomaly injection
                trace, types = introduce_anomaly(trace, single=False)
                for e_idx in range(len(trace)):
                    test_events.append(",".join([str(x) for x in trace[e_idx]]) + "," + str(key) + ",1,\"" + str(types) + "\"")

    with open(train_file, "w") as fout:
        fout.write(",".join(["Time", "Activity", "Resource", "Weekday", "Case", "Anomaly", "Type"]) + "\n")
        for e in train_events:
            fout.write(e + "\n")

    with open(test_file, "w") as fout:
        fout.write(",".join(["Time", "Activity", "Resource", "Weekday", "Case", "Anomaly", "Type"]) + "\n")
        for e in test_events:
            fout.write(e + "\n")


def introduce_anomaly(trace, single = False):
    """
    Add anomaly to the input trace

    :param trace: input trace, containing no anomalies
    :return: trace containing anomalies
    """
    def alter_activity_order(trace):
        if len(trace) == 1:
            return trace
        alter = random.randint(0, len(trace) - 2)
        tmp = trace[alter]
        trace[alter] = trace[alter + 1]
        trace[alter + 1] = tmp
        return trace

    def new_activity(trace):
        new_trace = []
        insert = random.randint(0, len(trace) - 1)
        for i in range(0, len(trace) + 1):
            if i < insert:
                new_trace.append(trace[i])
            elif i == insert:
                new_trace.append(trace[i-1][:])
                new_trace[-1][ACTIVITY_INDEX] = "a_NEW_ACTIVITY"
            else:
                new_trace.append(trace[i-1])
        return new_trace

    def new_date(trace):
        alter = random.randint(0, len(trace) - 1)
        new_date_generated = trace[alter][2]
        while new_date_generated == trace[alter][2]:
            new_date_generated = random.randint(0, 7)
        trace[alter][WEEKDAY_INDEX] = "wd_" + str(new_date_generated)
        return trace

    def new_resource(trace):
        alter = random.randint(0, len(trace) - 1)
        trace[alter][RESOURCE_INDEX] = "r_NEW_RESOURCE"
        return trace

    def generate_Anomay_single(trace):
        anomaly = random.randint(0,3)
        anomaly_types = []

        if anomaly == 0:
            trace = alter_activity_order(trace)
            anomaly_types.append("alter_order")
        elif anomaly == 1:
            trace = new_activity(trace)
            anomaly_types.append("new_activity")
        elif anomaly == 2:
            trace = new_date(trace)
            anomaly_types.append("new_date")
        elif anomaly == 3:
            trace = new_resource(trace)
            anomaly_types.append("new_resource")

        return (trace, anomaly_types)

    def generate_Anomaly(trace, num_diff_anoms, from_nums, to_nums):
        anoms = set()
        anomaly_types = []
        for i in range(num_diff_anoms):
            anomaly = random.randint(0,3)
            while anomaly in anoms: # Ensure each type of anomaly is only choosen once
                anomaly = random.randint(0,3)
            #for j in range(from_nums, to_nums + 1):
            for j in range(random.randint(from_nums, to_nums)):
                if anomaly == 0:
                    trace = alter_activity_order(trace)
                    anomaly_types.append("alter_order")
                elif anomaly == 1:
                    trace = new_activity(trace)
                    anomaly_types.append("new_activity")
                elif anomaly == 2:
                    trace = new_date(trace)
                    anomaly_types.append("new_date")
                elif anomaly == 3:
                    trace = new_resource(trace)
                    anomaly_types.append("new_resource")
        return (trace, anomaly_types)

    if single:
        return generate_Anomay_single(trace)
    else:
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



def preProcessData(path_to_data):
    for i in range(1,6):
        train = path_to_data + "BPIC15_train_" + str(i) + ".csv"
        test = path_to_data + "BPIC15_test_" + str(i) + ".csv"
        write_to_file(train, test, read_raw_file(path_to_data + "BPIC15_" + str(i) + "_sorted_new.csv"))

def preProcessData_total(path_to_data):
    total_log = {}
    test_keys = set()
    for i in range(5,0,-1):
        log = read_raw_file(path_to_data + "BPIC15_" + str(i) + "_sorted_new.csv")
        test_keys = test_keys.union(log.keys())
        total_log.update(log)

    train = path_to_data + "BPIC15_train_total.csv"
    test = path_to_data + "BPIC15_test_total.csv"
    write_to_file(train, test, total_log)
