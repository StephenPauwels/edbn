import pandas as pd
import extended_Dynamic_Bayesian_Network as edbn

import numpy as np
from dateutil.parser import parse
import matplotlib.pyplot as plt
import math

from LogFile import LogFile
import datetime
from sklearn.base import BaseEstimator

from scipy.spatial import ConvexHull

def analyse_timing_between_attributes():
    train = LogFile("../Data/bpic15_1_train_no_duration.csv", ",", 0, 500000, "Complete_Timestamp", "Case_ID", activity_attr="Activity")
    test = LogFile("../Data/bpic15_1_test_no_duration.csv", ",", 0, 500000, "Complete_Timestamp", "Case_ID", activity_attr="Activity")

    train.create_k_context()
    train.add_duration_to_k_context()

    test.create_k_context()
    test.add_duration_to_k_context()

    plt.hist(train.contextdata[train.contextdata["duration_0"] != 0]["duration_0"], bins=100)
    plt.show()

    #plt.hist(test.contextdata["duration_0"], bins=100)
    #plt.show()

    print(train.contextdata["duration_0"])
    grouped_training = train.contextdata.groupby(["Activity_Prev0", "Activity"])
    grouped_test = test.contextdata.groupby(["Activity_Prev0", "Activity"])

    print("Zero:", len(train.contextdata[train.contextdata["duration_0"] == 0]))
    print("Non-zero:", len(train.contextdata[train.contextdata["duration_0"] != 0]))

    """
    diff = []
    always = 0
    sometimes = 0
    never = 0

    for group in grouped_training:
        group_training = group[1]
        if group[0] in grouped_test.groups:
            #group_test = grouped_test.get_group(group[0])
            durations = group_training["duration_0"]
            #durations_test = group_test["duration_0"]
            #diff.append(math.fabs(np.mean(durations) - np.mean(durations_test)))
            nonzero = np.count_nonzero(durations)
            total = len(durations)
            if total > 0:
                ratio = nonzero / total
                print("Ratio nonzeros:", ratio)
                if ratio == 0:
                    always += 1
                elif ratio == 1:
                    never += 1
                else:
                    sometimes += 1

    print(always, sometimes, never)
    #print(np.mean(diff))
    """


def test():

    data = LogFile("../Data/BPIC15_train_1.csv", ",", 0, 500000, "Time", "Case", activity_attr="Activity")
    print(data.data["Weekday"])
    print(np.unique(data.data["Weekday"]))
    print(len(np.unique(data.data["Weekday"])))


def check_full():
    train_data = LogFile("../Data/bpic15_1_train.csv", ",", 0, 500000, "Complete_Timestamp", "Case_ID", activity_attr="Activity")
    d = train_data.data["case_IDofConceptCase"]
    print(d.value_counts())


    # df = pd.read_csv("../Data/bpic15_1_train.csv", header=0, nrows=500000, delimiter=",", dtype="str")
    # vals = np.unique(df["case_parts"].values)
    #
    # df2 = pd.read_csv("../Data/bpic15_1_test.csv", header=0, nrows=500000, delimiter=",", dtype="str")
    # vals2 = np.unique(df2["case_parts"].values.astype("str"))
    #
    # print(vals)
    # print(vals2)
    # print(np.setdiff1d(vals2, vals))

def checking_ouput_files():
    results_without = []
    with open("../Data/output2_1.csv") as finn:
        for line in finn:
            splitted = line.split(",")
            results_without.append((int(splitted[0]), float(splitted[1]), eval(splitted[2])))

    results_with = []
    with open("../Data/output_1.csv") as finn:
        for line in finn:
            splitted = line.split(",")
            results_with.append((int(splitted[0]), float(splitted[1]), eval(splitted[2])))

    for i in range(len(results_without)):
        for j in range(len(results_with)):
            if results_with[j][0] == results_without[i][0]:
                break

        print(results_without[i][0], i, "(", results_without[i][1], ")", j, "(", results_with[j][1], ")", results_without[i][2])
        input()


def test_file():
    train_file = "../Data/bpic15_1_train.csv"
    test_file = "../Data/bpic15_1_test.csv"

    train_data = LogFile(train_file, ",", 0, 500000, time_attr="Complete_Timestamp", trace_attr="Case_ID",
                         activity_attr="Activity")

    test_data = LogFile(test_file, ",", header=0, rows=500000, time_attr="Complete_Timestamp", trace_attr="Case_ID",
                        values=train_data.values)
    test_data.create_k_context()
    test_data.add_duration_to_k_context()
    print(test_data.contextdata["duration_0"].value_counts())

def visualize_durations():
    train_file = "../Data/bpic15_1_train.csv"
    test_file = "../Data/bpic15_1_test.csv"

    train_data = LogFile(train_file, ",", 0, 500000, time_attr="Complete_Timestamp", trace_attr="Case_ID",
                         activity_attr="Activity")
    train_data.create_k_context()
    train_data.add_duration_to_k_context()
    train_durations = []
    for row in train_data.contextdata.itertuples():
        train_durations.append(getattr(row, "duration_0"))
    train_durations = np.asarray(train_durations).reshape(-1,1)

    test_data = LogFile(test_file, ",", header=0, rows=500000, time_attr="Complete_Timestamp", trace_attr="Case_ID",
                        values=train_data.values)
    test_data.create_k_context()
    test_data.add_duration_to_k_context()
    normal_durations = []
    anom_durations = []
    test_durations = []
    for row in test_data.contextdata.itertuples():
        if test_data.convert_int2string("Anomaly", getattr(row, "Anomaly")) != "0":
            if "alter_duration" in test_data.convert_int2string("anom_types", getattr(row, "anom_types")):
                anom_durations.append(getattr(row, "duration_0"))
            else:
                normal_durations.append(getattr(row, "duration_0"))
        else:
            normal_durations.append(getattr(row, "duration_0"))
        test_durations.append(getattr(row, "duration_0"))
    normal_durations = [x for x in normal_durations if x >= 0]
    normal_durations = np.asarray(normal_durations).reshape(-1,1)
    anom_durations = np.asarray(anom_durations).reshape(-1,1)

    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(novelty=True).fit(train_durations)
    normal_lof = lof.predict(normal_durations)
    print(len([x for x in normal_lof if x == 1]), len([x for x in normal_lof if x == -1]))
    anom_lof = lof.predict(anom_durations)
    print(len([x for x in anom_lof if x == 1]), len([x for x in anom_lof if x == -1]))

    print(np.min(train_durations), np.mean(train_durations), np.max(train_durations), len(np.unique(train_durations)))
    print(np.min(normal_durations), np.mean(normal_durations), np.max(normal_durations))
    print(np.min(anom_durations), np.mean(anom_durations), np.max(anom_durations))
    print(np.min(test_durations), np.mean(test_durations), np.max(test_durations), len(np.unique(test_durations)))
 #   print(sorted(normal_durations))

    test_unique = sorted(np.unique(test_durations))
    y = [1] * len(test_unique)
    plt.plot(test_unique[:-3000], y[:-3000], "x")
    plt.show()

    print(len(train_data.contextdata.groupby(["Activity_Prev0", "Activity"])))
    print(len(test_data.contextdata.groupby(["Activity_Prev0", "Activity"])))


def read_and_save_durations():
    train_file = "../Data/bpic15_1_train_only_duration_1000_100.csv"
    test_file = "../Data/bpic15_1_test_only_duration_1000_100.csv"
    train_data = LogFile(train_file, ",", 0, 500000, time_attr="Complete_Timestamp", trace_attr="Case_ID",
                         activity_attr="Activity")
    test_data = LogFile(test_file, ",", header=0, rows=500000, time_attr="Complete_Timestamp", trace_attr="Case_ID",
                    values=train_data.values)

    train_data.create_k_context()
    train_data.add_duration_to_k_context()

    test_data.create_k_context()
    test_data.add_duration_to_k_context()

    vals = sorted(train_data.contextdata["duration_0"].values)
    vals = vals
    np.save("../Data/bpic15_1_train_only_duration_1000_100.npy", vals)

    vals = sorted(test_data.contextdata["duration_0"].values)
    vals = vals
    np.save("../Data/bpic15_1_test_only_duration_1000_100.npy", vals)


    groups = test_data.contextdata.groupby("Anomaly")
    i = 0
    for group in groups:
        durs = group[1]["duration_0"].values
        plt.plot(durs, [i] * len(durs))
        i += 1
        print(group)
    plt.show()


def plot_summaries():
    total_file = "../Data/BPIC15_1_sorted_new.csv"
    total_log = LogFile(total_file, ",", 0, 500000, time_attr="Complete Timestamp", trace_attr="Case ID",activity_attr="Activity")
    total_log.create_k_context()
    total_log.add_duration_to_k_context()

    train_log = LogFile("../Data/bpic15_1_train.csv", ",", 0, 500000, time_attr="Complete_Timestamp", trace_attr="Case_ID", activity_attr="Activity")
    train_log.create_k_context()
    train_log.add_duration_to_k_context()

    test_log = LogFile("../Data/bpic15_1_test.csv", ",", 0, 500000, time_attr="Complete_Timestamp", trace_attr="Case_ID", activity_attr="Activity", values=train_log.values)
    test_log.create_k_context()
    test_log.add_duration_to_k_context()

    total_durations = total_log.contextdata["duration_0"].values

    train_durations = train_log.contextdata["duration_0"].values
    test_durations = test_log.contextdata["duration_0"].values

    activity_var = edbn.Discrete_Variable("Activity", 0, 0, 0)
    duration_var = edbn.Continuous_Variable("duration_0", 1)
#    duration_var.add_parent(activity_var)

    duration_var.train(train_log.contextdata)
    train_scores = []
    train_scores_x = []
    test_scores = []
    test_scores_x = []
    test_scores_anoms = []
    test_scores_anoms_x = []

    changed_int = test_log.convert_string2int("time_anomaly", "Changed")
    x_log = True

    for row in train_log.contextdata.itertuples():
        train_scores.append(np.log(duration_var.test(row)))
        if x_log:
            train_scores_x.append(np.log(getattr(row, "duration_0")))
        else:
            train_scores_x.append(getattr(row, "duration_0"))

    score_zero_durations_anom = []
    score_zero_durations = []

    left_right_diff = []
    left_right_diff_anom = []
    for row in test_log.contextdata.itertuples():
        score = duration_var.test(row)
        score_right = duration_var.test_value(row, "right")
        if getattr(row, "time_anomaly") == changed_int:
            left_right_diff_anom.append(score - score_right)
            if score == 0:
                test_scores_anoms.append(-15)
                score_zero_durations_anom.append(getattr(row, "duration_0"))
            else:
                test_scores_anoms.append(np.log(score))
            if x_log:
                test_scores_anoms_x.append(np.log(getattr(row, "duration_0")))
            else:
                test_scores_anoms_x.append(getattr(row, "duration_0"))
        else:
            left_right_diff.append(score - score_right)

            if score == 0:
                test_scores.append(-15)
                score_zero_durations.append(getattr(row, "duration_0"))
            else:
                test_scores.append(np.log(score))
            if x_log:
                test_scores_x.append(np.log(getattr(row, "duration_0")))
            else:
                test_scores_x.append(getattr(row, "duration_0"))

    """
    print("Anom results:")
    score_zero_durations_anom.sort()
    print(score_zero_durations_anom[0], score_zero_durations_anom[-1], score_zero_durations_anom)
    print(np.log(score_zero_durations_anom[0]), np.log(score_zero_durations_anom[-1]), score_zero_durations_anom)

    print("Test results:")
    score_zero_durations.sort()
    print(score_zero_durations[0], score_zero_durations[-1], score_zero_durations)
    print(np.log(score_zero_durations[0]), np.log(score_zero_durations[-1]), np.log(np.asarray(score_zero_durations)))

    print("Anom results")
    left_right_diff_anom.sort()
    left_right_diff_anom = np.asarray(left_right_diff_anom)
    print(left_right_diff_anom[left_right_diff_anom > 0])

    print("Test results")
    left_right_diff.sort()
    left_right_diff = np.asarray(left_right_diff)
    print(left_right_diff[left_right_diff > 0])
    """
    left_right_diff_anom.sort()
    left_right_diff.sort()
    print(left_right_diff_anom)
    print(left_right_diff)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("log-duration")
    ax1.set_ylabel("histogram")
    ax1.hist(np.log(total_durations[total_durations > 0]), alpha=0.1)
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(train_scores_x, np.asarray(train_scores) + 50, "o", alpha=0.05)
    ax2.plot(test_scores_x, np.asarray(test_scores) + 25, "o", alpha=0.05)
    ax2.plot(test_scores_anoms_x, np.asarray(test_scores_anoms), "x")

    fig.tight_layout()
    plt.title("Analyse")
    plt.show()

    duration_type = test_log.convert_string2int("anom_types", "['alter_duration']")
    grouped_by_type = test_log.contextdata.groupby("anom_types")
    for group in grouped_by_type:
        if group[0] == duration_type:
            case_grouped = group[1].groupby("Case_ID")
            print(len(case_grouped))


    """
    train_durations = np.load(base + "train" + file + ".npy")
    test_durations = np.load(base + "test" + file + "_normal_durations.npy")
    next_anoms = np.load(base + "test" + file + "_next_anom_durations.npy")
    anom_durations = np.load(base + "test" + file + "_anom_durations.npy")

    k_dist_train = k_distance(20).fit(train_durations)
    test_durations_scores = k_dist_train.score_distance(test_durations)
    next_anoms_scores = k_dist_train.score_distance((next_anoms))
    anom_durations_scores = k_dist_train.score_distance((anom_durations))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("log-duration")
    ax1.set_ylabel("histogram")
    ax1.hist(np.log([d for d in total_log.contextdata["duration_0"].values if d > 0]), alpha=0.1)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(np.log(test_durations), np.log(test_durations_scores), 'o', alpha=0.1, aa=True)
    ax2.plot(np.log(next_anoms), np.log(next_anoms_scores), 'x', alpha=0.4, aa=True)
    ax2.plot(np.log(anom_durations), np.log(anom_durations_scores), 'x', aa=True)
    ax2.tick_params(axis='y')

    fig.tight_layout()
    plt.title(file)
    plt.show()
    """

def analyze_k_dist():
    total_file = "../Data/BPIC15_1_sorted_new.csv"
    log = LogFile(total_file, ",", 0, 500000, time_attr="Complete Timestamp", trace_attr="Case ID",activity_attr="Activity")
    log.create_k_context()
    log.add_duration_to_k_context()


    train_durations = np.load("../Data/bpic15_1_train_durations.npy")
    total_test_durations = np.load("../Data/bpic15_1_test_durations.npy")
    test_durations = np.load("../Data/bpic15_1_test_normal_test_durations.npy")
    anom_durations = sorted(np.load("../Data/bpic15_1_test_anom_durations.npy"))
    next_anoms = sorted(np.load("../Data/bpic15_1_test_next_anom_durations.npy"))
    #train_durations = train_durations[int(len(train_durations)*0.05):int(len(train_durations)*0.95)]

    print(train_durations)
    print(test_durations)

    k_dist_train = k_distance(1).fit(train_durations)
    k_dist_test = k_distance(5).fit(test_durations)

    print(k_dist_train.cdf_x_convex)
    print(k_dist_train.cdf_y_convex)
    for i in range(len(k_dist_train.cdf_y_convex)-1):
        print(k_dist_train.cdf_y_convex[i+1] - k_dist_train.cdf_y_convex[i])
    plt.plot(k_dist_train.cdf_x, k_dist_train.cdf_y, "x")
    plt.plot(k_dist_train.cdf_x, k_dist_train.cdf_y, lw=4)
    plt.plot(k_dist_train.cdf_x_convex, k_dist_train.cdf_y_convex, "--")
    plt.show()

    plt.plot(k_dist_train.cdf_x, k_dist_train.cdf_y)
    plt.show()

    plt.plot(k_dist_train.pdf_x, k_dist_train.pdf_y)
    plt.show()

    min_duration = min(anom_durations)
    max_duration = max(anom_durations)
    #filtered_durations = [d for d in test_durations if d > min_duration and d < max_duration and d not in anom_durations]
    #filtered_next_durations = [d for d in next_anoms if d > min_duration and d < max_duration and d not in anom_durations]

    plt.plot(np.log(train_durations), np.log(k_dist_train.score_distance(train_durations)), "o")
    plt.plot(np.log(test_durations), np.log(k_dist_train.score_distance(test_durations)), "x")
    plt.plot(np.log(anom_durations), np.log(k_dist_train.score_distance(anom_durations)), "x")
    plt.show()

    plt.plot(np.log(train_durations), np.log(k_dist_train.score_samples(train_durations)), "o")
    plt.plot(np.log(test_durations), np.log(k_dist_train.score_samples(test_durations)), "x")
    plt.plot(np.log(anom_durations), np.log(k_dist_train.score_samples(anom_durations)), "x")
    plt.show()

    min_train = min(train_durations)
    max_train = max(train_durations)
    x_vals = np.linspace(min_train, max_train, 100000)
    plt.plot(np.log(x_vals), np.log(k_dist_train.score_samples(x_vals)), "o")
    plt.show()

    test_durations_scores = k_dist_train.score_samples(test_durations)
    next_anoms_scores = k_dist_train.score_samples((next_anoms))
    anom_durations_scores = k_dist_train.score_samples((anom_durations))

    print(len([d for d in test_durations_scores if d > np.power(np.e, -2)]), len(test_durations_scores))
    print(len([d for d in anom_durations_scores if d > np.power(np.e, -2)]), len(anom_durations_scores))
    print(len([d for d in anom_durations_scores if d > np.power(np.e, -10)]), len(anom_durations_scores))
    print(len([d for d in next_anoms if d > np.power(np.e, -2)]), len(next_anoms))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("log-duration")
    ax1.set_ylabel("histogram")
    ax1.hist(np.log([d for d in log.contextdata["duration_0"].values if d > 0]), alpha=0.1)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(np.log(test_durations), np.log(test_durations_scores), 'o', alpha=0.1, aa=True)
    ax2.plot(np.log(next_anoms), np.log(next_anoms_scores), 'x', alpha=0.4, aa=True)
    ax2.plot(np.log(anom_durations), np.log(anom_durations_scores), 'x', aa=True)
    ax2.tick_params(axis='y')

    fig.tight_layout()
    plt.show()

    log_durations = log.contextdata["duration_0"].values
    plt.plot(np.log(log_durations), np.log(k_dist_train.score_samples(log_durations)), 'o', alpha=0.1)
    plt.show()

    plt.plot(np.log(test_durations), np.log(k_dist_train.score_distance(test_durations)), 'o', alpha=0.1, aa=True)
    plt.plot(np.log(next_anoms), np.log(k_dist_train.score_distance(next_anoms)), 'x', alpha=0.4, aa=True)
  #  plt.plot(np.log(anom_durations), np.log(k_dist_train.score_distance(anom_durations)), 'x', aa=True)
    plt.show()

    plt.hist(test_durations)
    plt.hist(next_anoms, alpha=0.7)
    plt.hist(anom_durations, alpha=0.4)
    plt.show()

    plt.plot(train_durations, np.log(k_dist_train.score_samples(train_durations)))
    plt.plot(train_durations, [0] * len(train_durations), "x")
    plt.show()

    plt.plot(test_durations, np.log(k_dist_train.score_samples(test_durations)))
#    plt.plot(train_durations, [1] * len(train_durations), "x")
    plt.plot(test_durations, [0] * len(test_durations), "x")
    plt.show()
    print(k_dist_train.score())

    train_dist = k_dist_train.get_k_neighbour(test_durations)
    distances = sorted(np.abs(train_dist - test_durations))

    plt.plot(list(range(len(distances))), np.log(distances))
    plt.show()

    #plt.plot(list(range(len(k_dist_train.distances))), k_dist_train.distances)
    #plt.show()

    #plt.plot(train_dist, k_dist_train.score_distance(train_dist))
    #plt.show()

def check_anom_timings():
    train_file = "../Data/bpic15_1_train_only_duration_1000_100.csv"
    test_file = "../Data/bpic15_1_test_only_duration_1000_100.csv"
    train_data = LogFile(train_file, ",", 0, 500000, time_attr="Complete_Timestamp", trace_attr="Case_ID",
                         activity_attr="Activity")
    test_data = LogFile(test_file, ",", header=0, rows=500000, time_attr="Complete_Timestamp", trace_attr="Case_ID",
                    values=train_data.values)

    train_data.create_k_context()
    train_data.add_duration_to_k_context()

    test_data.create_k_context()
    test_data.add_duration_to_k_context()

    changed_int = test_data.convert_string2int("time_anomaly", "Changed")
    not_changed_int = test_data.convert_string2int("time_anomaly", "nan")
    anom_durations = test_data.contextdata[(test_data.contextdata['time_anomaly'] == changed_int) & (test_data.contextdata['time_anomaly_Prev0'] == not_changed_int)]["duration_0"].values
    np.save("../Data/bpic15_1_test_only_duration_1000_100_anom_durations.npy", anom_durations)
    durations = test_data.contextdata["duration_0"].values

    next_anom_durations = test_data.contextdata[(test_data.contextdata['time_anomaly'] == changed_int) & (test_data.contextdata['time_anomaly_Prev0'] == changed_int)]["duration_0"].values
    np.save("../Data/bpic15_1_test_only_duration_1000_100_next_anom_durations.npy", next_anom_durations)

    normal_test_durations = test_data.contextdata[(test_data.contextdata['time_anomaly'] == not_changed_int) & (test_data.contextdata['time_anomaly_Prev0'] == not_changed_int)]["duration_0"].values
    np.save("../Data/bpic15_1_test_only_duration_1000_100_normal_durations.npy", normal_test_durations)

    min_duration = min(anom_durations)
    max_duration = max(anom_durations)
    filtered_durations = [d for d in durations if d > min_duration and d < max_duration]

    plt.plot(np.log(filtered_durations), [0] * len(filtered_durations), "o")
    plt.plot(np.log(anom_durations), [0] * len(anom_durations), "x")
    plt.show()



class k_distance(BaseEstimator):

    def __init__(self, k=5):
        self.k = k

    def fit(self, X):
        self.values = sorted(X)

        """
        x,y = self.get_cumm_values()

        x = [-1] + x
        y = [0] + y

        self.cdf_x = x
        self.cdf_y = y

        #self.cdf_x = [self.cdf_x[-1]] + self.cdf_x[:-1]
        #self.cdf_y = [self.cdf_y[-1]] + self.cdf_y[:-1]

        cdf_points = np.asarray(list(zip(self.cdf_x, self.cdf_y)))
        if len(cdf_points) > 4:
            hull = ConvexHull(cdf_points)
            self.cdf_x_convex = cdf_points[hull.vertices,0]
            self.cdf_y_convex = cdf_points[hull.vertices,1]
        else:
            self.cdf_x_convex = self.cdf_x
            self.cdf_y_convex = self.cdf_y

        x0 = self.cdf_x_convex[0]
        y0 = self.cdf_y_convex[0]

        self.cdf_x_convex = np.flip(np.append(self.cdf_x_convex[1:], x0))
        self.cdf_y_convex = np.flip(np.append(self.cdf_y_convex[1:], y0))

        self.cdf_x = self.cdf_x_convex
        self.cdf_y = self.cdf_y_convex

        self.pdf_x = self.cdf_x[1:]
        self.pdf_y = np.diff(self.cdf_y) / np.diff(self.cdf_x)
        """
#        self.distances = sorted(np.abs(np.asarray(self.get_k_neighbour(self.values)) - np.asarray(self.values)))
        self.distances = sorted(self.get_k_neighbour(self.values))

        return self

    def score_samples(self, Y):
        distances = self.get_k_neighbour(Y)
        indexes = np.searchsorted(self.pdf_x, distances, side='left')
        results = []
        min_p = min(self.pdf_y)
        for i in indexes:
            if i == len(self.pdf_y):
                results.append(min_p)
            else:
                results.append(self.pdf_y[i])
        return results

    def score_distance(self, Y):
        neighbours = self.get_k_neighbour(Y)
        indexes = np.searchsorted(self.distances, neighbours, side="left")
        return 1 - (indexes / len(self.distances))

    def score(self):
        return np.sum(np.log(self.score_samples(self.values)))

    def get_k_neighbour(self, Y):
        dist = []
        indexes = np.searchsorted(self.values, Y)
        for i in range(len(indexes)):
            begin = max(0, indexes[i] - self.k - 1)
            end = min(len(self.values), indexes[i] + self.k + 1)
            if indexes[i] < len(self.values) and self.values[indexes[i]] == Y[i]:
                neighbours = self.values[begin:indexes[i]] + self.values[indexes[i] + 1:end]
            else:
                neighbours = self.values[begin:end]
            dist.append(sorted(np.abs(np.asarray(neighbours) - Y[i]))[min(self.k - 1, len(neighbours) - 1)])
        return dist

    def get_cumm_values(self):
        dists = sorted(self.get_k_neighbour(self.values))
        unique, counts = np.unique(dists, return_counts=True)
        total_seen = 0
        x = []
        y = []
        for i in range(len(unique)):
            total_seen += counts[i]
            y.append(total_seen / len(self.values))
            x.append(unique[i])
        return x, y



if __name__ == "__main__":
    #analyse_timing_between_attributes()
    #test()
    #check_full()
    #checking_ouput_files()
    #test_file()
    #visualize_durations()
    #read_and_save_durations()
    #analyze_k_dist()
    #check_anom_timings()

    plot_summaries()