import pandas as pd
import os
import pickle

from Utils.LogFile import LogFile
from Camargo.support_modules.role_discovery import role_discovery

BPIC15 = "BPIC15"
BPIC12 = "BPIC12"
BPIC12W = "BPIC12W"
HELPDESK = "HELPDESK"
CLICKS = "CLICKS"
BPIC14 = "BPIC14"
BPIC18 = "BPIC18"

LOGFILE_PATH = "../Data/Logfiles"

def preprocess(logfile, add_end, reduce_tasks, resource_pools, resource_attr, remove_resource):
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

    # Discover Roles
    if resource_pools and resource_attr is not None:
        resources, resource_table = role_discovery(logfile.get_data(), resource_attr, 0.5)
        log_df_resources = pd.DataFrame.from_records(resource_table)
        log_df_resources = log_df_resources.rename(index=str, columns={"resource": resource_attr})

        logfile.data = logfile.data.merge(log_df_resources, on=resource_attr, how='left')
        logfile.categoricalAttributes.add("role")
        if remove_resource:
            logfile.data = logfile.data.drop([resource_attr], axis=1)

        resource_attr = "role"

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
    cache_file = LOGFILE_PATH + "/" + "_".join(filename_parts)

    colTitles = []

    if os.path.exists(cache_file):
        print("Loading file from cache")
        with open(cache_file, "rb") as pickle_file:
            preprocessed_log = pickle.load(pickle_file)
    else:
        resource_attr = None
        if dataset == BPIC15:
            logfile = LogFile("../Data/BPIC15_1_sorted_new.csv", ",", 0, dataset_size, "Complete Timestamp", "Case ID", activity_attr="Activity", convert=False, k=k)
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
        elif dataset == CLICKS:
            logfile = LogFile("../Data/BPI2016_Clicks_Logged_In.csv", ";", 0, dataset_size, "TIMESTAMP", "SessionID", activity_attr="URL_FILE", convert=False, k=k)
            #logfile.keep_attributes(["SessionID", "URL_FILE", "REF_URL_category"])
            logfile.keep_attributes(["SessionID", "URL_FILE"])
            logfile.filter_case_length(5)
        elif dataset == BPIC14:
            logfile = LogFile("../Data/BPIC2014.csv", ";", 0, dataset_size, "Actual Start", "Change ID", activity_attr="Change Type", convert=False, k=k)
            logfile.keep_attributes(["Change ID", "Change Type"])
        elif dataset == BPIC18:
            logfile = LogFile("../Data/bpic2018.csv", ",", 0, dataset_size, "startTime", "case", activity_attr="event", convert=False, k=k)
            logfile.keep_attributes(["case", "event", "subprocess" ])
        else:
            print("Unknown Dataset")
            return None
        preprocessed_log = preprocess(logfile, add_end, reduce_tasks, resource_pools, resource_attr, remove_resource)

        preprocessed_log.data = preprocessed_log.data.reindex(columns=colTitles)

        preprocessed_log.create_k_context()
        with open(cache_file, "wb") as pickle_file:
            pickle.dump(preprocessed_log, pickle_file)
    return preprocessed_log, "_".join(filename_parts)