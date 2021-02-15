from Utils.LogFile import LogFile
from data.data import Data

all_data = {"Helpdesk": "../Data/Helpdesk.csv",
            "BPIC12": "../Data/BPIC12.csv",
            "BPIC12W": "../Data/BPIC12W.csv",
            "BPIC15_1": "../Data/BPIC15_1_sorted_new.csv",
            "BPIC15_2": "../Data/BPIC15_2_sorted_new.csv",
            "BPIC15_3": "../Data/BPIC15_3_sorted_new.csv",
            "BPIC15_4": "../Data/BPIC15_4_sorted_new.csv",
            "BPIC15_5": "../Data/BPIC15_5_sorted_new.csv",
            "BPIC18": "../Data/BPIC18.csv",
            "BPIC17": "../Data/bpic17_test.csv",
            "BPIC19": "../Data/BPIC19.csv",
            "BPIC11": "../Data/BPIC11.csv",
            "SEPSIS": "../Data/Sepsis.csv"}


def get_data(data_name, sep=",", time="completeTime", case="case", activity="event", resource="role"):
    if data_name in all_data:
        d = Data(data_name, LogFile(all_data[data_name], sep, 0, None, time, case, activity_attr=activity, convert=False))
        d.logfile.keep_attributes([activity, resource, time])
        return d
    return None


def get_all_data():
    datasets = []
    for d in all_data:
        if d != "BPIC18":
            datasets.append(get_data(d))
    return datasets
