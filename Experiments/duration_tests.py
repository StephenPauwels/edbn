import matplotlib.pyplot as plt
import extended_Dynamic_Bayesian_Network as edbn
import eDBN.Execute as edbn_exec
import numpy as np
import Utils.PlotResults as plot

from LogFile import LogFile


def analyze():
    data_file = "../Data/synth_duration/data.csv"
    log = LogFile(data_file, ",", 0, 500000, time_attr="date", trace_attr="trace",activity_attr="event")
    log.create_k_context()
    log.add_duration_to_k_context()

    test_file = "../Data/synth_duration/data_test.csv"
    test_log = LogFile(test_file, ",", 0, 500000, time_attr="date", trace_attr="trace",activity_attr="event")
    test_log.create_k_context()
    test_log.add_duration_to_k_context()
    test_log.contextdata.to_csv("../Data/synth_duration/contextdata.csv", index=False)

    print(test_log.contextdata["duration_0"].value_counts())

    ax1 = log.contextdata["duration_0"].hist(alpha=0.05, grid=False)

    duration_var = edbn.Continuous_Variable("duration_0", 1)
    activity_var = edbn.Discrete_Variable("event", 0, 0, 0)
    duration_var.add_parent(activity_var)

    duration_var.train(log.contextdata)

    train_scores = []
    train_x = []
    for row in log.contextdata.itertuples():
        score = duration_var.test(row)
        if score <= 0:
            train_scores.append(-15)
        else:
            train_scores.append(np.log(duration_var.test(row)))
        train_x.append(getattr(row, "duration_0"))

    ax2 = ax1.twinx()
    ax2.plot(train_x, train_scores, "x", alpha=0.5, color="orange")
    plt.show()



    ax1 = test_log.contextdata["duration_0"].hist(alpha=0.05, grid=False)

    test_scores = []
    test_x = []
    anom_scores = []
    anom_x = []
    for row in test_log.contextdata.itertuples():
        score = duration_var.test(row)
        if getattr(row, "changed") == test_log.convert_string2int("changed", "0"):
            if score <= 0:
                test_scores.append(-15)
            else:
                test_scores.append(np.log(score))
            test_x.append(getattr(row, "duration_0"))
        else:
            if score <= 0:
                anom_scores.append(-15)
            else:
                anom_scores.append(np.log(score))
            anom_x.append(getattr(row, "duration_0"))

    ax2 = ax1.twinx()
    ax2.plot(test_x, test_scores, "x", alpha=0.1, color="orange")
    ax2.plot(anom_x, anom_scores, "x", alpha=0.1, color="green")

    plt.show()

def run_anom():
    data_file = "../Data/synth_duration/data.csv"
    log = LogFile(data_file, ",", 0, 500000, time_attr="date", trace_attr="trace",activity_attr="event")
    log.create_k_context()
    log.add_duration_to_k_context()

    test_file = "../Data/synth_duration/data_test.csv"
    test_log = LogFile(test_file, ",", 0, 500000, time_attr="date", trace_attr="trace",activity_attr="event", values=log.values)
    test_log.create_k_context()
    test_log.add_duration_to_k_context()

    log.keep_attributes(["event", "date", "trace", "process", "resource", "random"])

    model = edbn_exec.train(log)
    edbn_exec.test(test_log, "../Data/synth_duration/synth_out.csv", model, label = "anomaly", normal_val = "0", train_data=log)

    plot.plot_single_roc_curve("../Data/synth_duration/synth_out.csv", title="Synth Data")
    plot.plot_single_prec_recall_curve("../Data/synth_duration/synth_out.csv", title="Synth Data")


if __name__ == "__main__":
    analyze()
    run_anom()
