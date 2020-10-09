from Utils.LogFile import LogFile
from EDBN.Execute import train as edbn_train
from Predictions.eDBN_Prediction import predict_next_event, predict_next_event_update, predict_next_event_multi

import matplotlib.pyplot as plt

def _test1():
    data = "../Data/BPIC15_1_sorted_new.csv"
    case_attr = "case"
    act_attr = "event"

    logfile = LogFile(data, ",", 0, None, None, case_attr,
                      activity_attr=act_attr, convert=False, k=5)
    logfile.keep_attributes(["case", "event", "role"])
    logfile.convert2int()
    # logfile.filter_case_length(5)

    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(70, case=True, method="train-test")

    model = edbn_train(train_log)
    acc = predict_next_event(model, test_log)
    acc_update = predict_next_event_update(model, test_log)
    print("ACC:", acc, acc_update)

if __name__ == "__main__":
    data = "../Data/BPIC15_1_sorted_new.csv"
    case_attr = "case"
    act_attr = "event"

    logfile = LogFile(data, ",", 0, None, "completeTime", case_attr,
                      activity_attr=act_attr, convert=False, k=5)
    logfile.keep_attributes(["case", "event", "role"])
    logfile.convert2int()
    logfile.create_k_context()

    weeks = logfile.split_days("%Y-%m-%d %H:%M:%S")
    weeks_sorted = sorted(weeks.keys())
    num_weeks = len(weeks_sorted)

    for i in range(num_weeks):
        weeks[weeks_sorted[i]]["model"] = edbn_train(weeks[weeks_sorted[i]]["data"])

    accs1 = []
    for i in range(1, num_weeks):
        accs1.append(predict_next_event_multi([weeks[w]["model"] for w in weeks_sorted[:i]], weeks[weeks_sorted[i]]["data"]))

    accs2 = []
    for i in range(1, num_weeks):
        accs2.append(predict_next_event(weeks[weeks_sorted[i-1]]["model"], weeks[weeks_sorted[i]]["data"]))

    accs3 = []
    for i in range(1, num_weeks):
        accs3.append(predict_next_event_multi([weeks[w]["model"] for w in weeks_sorted[:i]], weeks[weeks_sorted[i]]["data"], True))

    edbn_model = edbn_train(weeks[weeks_sorted[0]]["data"])
    accs4 = []
    for i in range(1, num_weeks):
        accs4.append(predict_next_event_update(edbn_model, weeks[weeks_sorted[i]]["data"]))

    plt.plot(weeks_sorted[1:num_weeks], accs1, "o")
    plt.plot(weeks_sorted[1:num_weeks], accs2, "x")
    plt.plot(weeks_sorted[1:num_weeks], accs3, "o")
    plt.plot(weeks_sorted[1:num_weeks], accs4, "+")

    plt.show()

    import numpy as np

    print("Accs (multi-weeks, no unknown):", np.average(accs1))
    print("Accs2 (prev-week):", np.average(accs2))
    print("Accs3 (multi-weeks, unknown):", np.average(accs3))
    print("Accs4 (update model):", np.average(accs4))