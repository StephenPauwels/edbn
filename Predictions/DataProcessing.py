"""
    Author: Stephen Pauwels
"""

import os
import pickle

import pandas as pd

from support_modules.role_discovery import role_discovery
from Utils.LogFile import LogFile

BPIC15 = "BPIC15"
BPIC15_1 = "BPIC15_1"
BPIC15_2 = "BPIC15_2"
BPIC15_3 = "BPIC15_3"
BPIC15_4 = "BPIC15_4"
BPIC15_5 = "BPIC15_5"
BPIC12 = "BPIC12"
BPIC12W = "BPIC12W"
HELPDESK = "HELPDESK"
BPIC18 = "BPIC18"

LOGFILE_PATH = "../Data/Logfiles"

def preprocess(logfile, add_end, reduce_tasks, resource_pools, resource_attr, remove_resource):
    # Discover Roles
    if resource_pools and resource_attr is not None:
        resources, resource_table = role_discovery(logfile.get_data(), resource_attr, 0.5)
        log_df_resources = pd.DataFrame.from_records(resource_table)
        log_df_resources = log_df_resources.rename(index=str, columns={"resource": resource_attr})
        print(logfile.data)
        logfile.data = logfile.data.merge(log_df_resources, on=resource_attr, how='left')
        logfile.categoricalAttributes.add("role")
        if remove_resource:
            logfile.data = logfile.data.drop([resource_attr], axis=1)

        resource_attr = "role"
    else:
        logfile.data = logfile.data.rename(columns={resource_attr: "role"})
        logfile.categoricalAttributes.add("role")

    print(logfile.data)

    if add_end:
        cases = logfile.get_cases()
        new_data = []
        for case_name, case in cases:
            record = {}
            for col in logfile.data:
                if col == logfile.trace:
                    record[col] = case_name
                else:
                    record[col] = "start"
            new_data.append(record)

            for i in range(0, len(case)):
                new_data.append(case.iloc[i].to_dict())

            record = {}
            for col in logfile.data:
                if col == logfile.trace:
                    record[col] = case_name
                else:
                    record[col] = "end"
            new_data.append(record)

        logfile.data = pd.DataFrame.from_records(new_data)

    # Check for dublicate events with same resource
    if reduce_tasks and resource_attr is not None:
        cases = logfile.get_cases()
        reduced = []
        for case_name, case in cases:
            reduced.append(case.iloc[0].to_dict())
            current_trace = [case.iloc[0][[logfile.activity, resource_attr]].values]
            for i in range(1, len(case)):
                if case.iloc[i][logfile.activity] == current_trace[-1][0] and \
                        case.iloc[i][resource_attr] == current_trace[-1][1]:
                    pass
                else:
                    current_trace.append(case.iloc[i][[logfile.activity, resource_attr]].values)
                    reduced.append(case.iloc[i].to_dict())
        logfile.data = pd.DataFrame.from_records(reduced)
        print("Removed duplicated events")

    logfile.convert2int()

    return logfile

def get_data(dataset, dataset_size, k, add_end, reduce_tasks, resource_pools, remove_resource):
    filename_parts = [dataset, str(dataset_size), str(k)]
    for v in [add_end, reduce_tasks, resource_pools, remove_resource]:
        if v:
            filename_parts.append(str(1))
        else:
            filename_parts.append(str(0))
    print(filename_parts)
    cache_file = LOGFILE_PATH + "/" + "_".join(filename_parts)

    colTitles = []

    if os.path.exists(cache_file):
        print("Loading file from cache")
        with open(cache_file, "rb") as pickle_file:
            preprocessed_log = pickle.load(pickle_file)
    else:
        resource_attr = None
        if dataset == BPIC15_1 or dataset == BPIC15:
            logfile = LogFile("../Data/BPIC15_1_sorted_new.csv", ",", 0, dataset_size, "Complete Timestamp", "Case ID", activity_attr="Activity", convert=False, k=k)
            resource_attr = "Resource"
            colTitles = ["Case ID", "Activity", "Resource"]
            logfile.keep_attributes(colTitles)
            logfile.filter_case_length(5)
        elif dataset == BPIC15_2:
            logfile = LogFile("../Data/BPIC15_2_sorted_new.csv", ",", 0, dataset_size, "Complete Timestamp", "Case ID",
                              activity_attr="Activity", convert=False, k=k)
            resource_attr = "Resource"
            colTitles = ["Case ID", "Activity", "Resource"]
            logfile.keep_attributes(colTitles)
            logfile.filter_case_length(5)
        elif dataset == BPIC15_3:
            logfile = LogFile("../Data/BPIC15_3_sorted_new.csv", ",", 0, dataset_size, "Complete Timestamp", "Case ID", activity_attr="Activity", convert=False, k=k)
            resource_attr = "Resource"
            colTitles = ["Case ID", "Activity", "Resource"]
            logfile.keep_attributes(colTitles)
            logfile.filter_case_length(5)
        elif dataset == BPIC15_4:
            logfile = LogFile("../Data/BPIC15_4_sorted_new.csv", ",", 0, dataset_size, "Complete Timestamp", "Case ID", activity_attr="Activity", convert=False, k=k)
            resource_attr = "Resource"
            colTitles = ["Case ID", "Activity", "Resource"]
            logfile.keep_attributes(colTitles)
            logfile.filter_case_length(5)
        elif dataset == BPIC15_5:
            logfile = LogFile("../Data/BPIC15_5_sorted_new.csv", ",", 0, dataset_size, "Complete Timestamp", "Case ID", activity_attr="Activity", convert=False, k=k)
            resource_attr = "Resource"
            colTitles = ["Case ID", "Activity", "Resource"]
            logfile.keep_attributes(colTitles)
            logfile.filter_case_length(5)
        elif dataset == BPIC12:
            logfile = LogFile("../Data/Camargo_BPIC2012.csv", ",", 0, dataset_size, "completeTime", "case", activity_attr="event", convert=False, k=k)
            resource_attr = "org:resource"
            colTitles = ["case", "event", "org:resource"]
            logfile.keep_attributes(colTitles)
            logfile.filter_case_length(5)
        elif dataset == BPIC12W:
            logfile = LogFile("../Data/Camargo_BPIC12W.csv", ",", 0, dataset_size, "completeTime", "case", activity_attr="event", convert=False, k=k)
            resource_attr = "org:resource"
            colTitles = ["case", "event", "org:resource"]
            logfile.keep_attributes(colTitles)
            logfile.filter_case_length(5)
        elif dataset == HELPDESK:
            logfile = LogFile("../Data/Camargo_Helpdesk.csv", ",", 0, dataset_size, "completeTime", "case", activity_attr="event", convert=False, k=k)
            resource_attr = "Resource"
            colTitles = ["case", "event", "Resource"]
            logfile.keep_attributes(colTitles)
            logfile.filter_case_length(3)
        elif dataset == BPIC18:
            logfile = LogFile("../Data/bpic2018.csv", ",", 0, dataset_size, "startTime", "case", activity_attr="event", convert=False, k=k)
            colTitles = ["case", "event", "subprocess"]
            logfile.keep_attributes(colTitles)
        else:
            print("Unknown Dataset")
            return None

        preprocessed_log = preprocess(logfile, add_end, reduce_tasks, resource_pools, resource_attr, remove_resource)

        preprocessed_log.create_k_context()
        with open(cache_file, "wb") as pickle_file:
            pickle.dump(preprocessed_log, pickle_file)
    return preprocessed_log, "_".join(filename_parts)

def calc_charact():
    import numpy as np
    print("Calculating characteristics")
    datasets = [BPIC12, BPIC12W, BPIC15_1, BPIC15_2, BPIC15_3, BPIC15_4, BPIC15_5, HELPDESK]
    for dataset in datasets:
        logfile, name = get_data(dataset, 20000000, 0, False, False, False, True)
        cases = logfile.get_cases()
        case_lengths = [len(c[1]) for c in cases]
        print("Logfile:", name)
        print("Num events:", len(logfile.get_data()))
        print("Num cases:", len(cases))
        print("Num activities:", len(logfile.get_data()[logfile.activity].unique()))
        print("Avg activities in case:", np.average(case_lengths))
        print("Max activities in case:", max(case_lengths))
        print()


if __name__ == "__main__":
    calc_charact()

