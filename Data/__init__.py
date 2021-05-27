from Utils.LogFile import LogFile
from Data.data import Data

BASE_FOLDER = "~/PycharmProjects/edbn/"

all_data = {"Helpdesk": BASE_FOLDER + "Data/Helpdesk.csv",
            "BPIC12": BASE_FOLDER + "Data/BPIC12.csv",
            "BPIC12W": BASE_FOLDER + "Data/BPIC12W.csv",
            "BPIC15_1": BASE_FOLDER + "Data/BPIC15_1_sorted_new.csv",
            "BPIC15_2": BASE_FOLDER + "Data/BPIC15_2_sorted_new.csv",
            "BPIC15_3": BASE_FOLDER + "Data/BPIC15_3_sorted_new.csv",
            "BPIC15_4": BASE_FOLDER + "Data/BPIC15_4_sorted_new.csv",
            "BPIC15_5": BASE_FOLDER + "Data/BPIC15_5_sorted_new.csv",
            "BPIC18": BASE_FOLDER + "Data/BPIC18.csv",
            "BPIC17": BASE_FOLDER + "Data/bpic17_test.csv",
            "BPIC19": BASE_FOLDER + "Data/BPIC19.csv",
            "BPIC11": BASE_FOLDER + "Data/BPIC11.csv",
            "SEPSIS": BASE_FOLDER + "Data/Sepsis.csv",
            "COSELOG_1": BASE_FOLDER + "Data/Coselog_1.csv",
            "COSELOG_2": BASE_FOLDER + "Data/Coselog_2.csv",
            "COSELOG_3": BASE_FOLDER + "Data/Coselog_3.csv",
            "COSELOG_4": BASE_FOLDER + "Data/Coselog_4.csv",
            "COSELOG_5": BASE_FOLDER + "Data/Coselog_5.csv",
            "Helpdesk2": BASE_FOLDER + "Data/helpdesk2.csv"}


def get_data(data_name, sep=",", time="completeTime", case="case", activity="event", resource="role"):
    if data_name in all_data:
        d = Data(data_name, LogFile(all_data[data_name], sep, 0, None, time, case, activity_attr=activity, convert=False))
        if resource:
            d.logfile.keep_attributes([activity, resource, time])
        else:
            d.logfile.keep_attributes([activity, time])
        return d
    print("ERROR: Datafile not found")
    print("ERROR: Possibilities:", ",".join(all_data.keys()))
    raise NotImplementedError


def get_all_data():
    datasets = []
    for d in all_data:
        if d != "BPIC18":
            datasets.append(get_data(d))
    return datasets
