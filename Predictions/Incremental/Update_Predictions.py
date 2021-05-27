import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from math import log

import setting
import data
import Methods
import metric
import Utils.LogFile as logfile


def store_results(file, results):
    with open(file, "w") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")


def dbn_adaptive_window(dataset):
    import copy

    d = data.get_data(dataset)
    s = copy.copy(setting.STANDARD)
    s.train_percentage = 50
    d.prepare(setting.STANDARD)
    d.create_batch("week", "%Y-%m-%d %H:%M:%S")

    dbn = Methods.get_prediction_method("DBN")

    batch_ids = d.get_batch_ids()

    results = []

    train_data = [d.get_test_batch(0)]
    model = dbn.train(logfile.combine(train_data))

    for i in range(1, len(batch_ids)):
        print(i, "/", len(batch_ids))

        ###
        # PREDICT
        ###
        predict_data = d.get_test_batch(i)
        results.extend(dbn.test(model, predict_data))


        ###
        # SELECT TRAIN DATA
        ###
        predict_trace_scores = [s.get_total_score() for s in model.test(predict_data.get_data())]

        updated_data = []
        max_p = 0
        max_j = 0
        p_vals = []
        for j in range(len(train_data)):
            test_log = train_data[j]
            scores = [s.get_total_score() for s in model.test(test_log.get_data())]
            p_vals.append(log(stats.ks_2samp(scores, predict_trace_scores).pvalue + 0.0001))

        avg_p = np.mean(p_vals)
        for j in range(len(p_vals)):
            if p_vals[j] >= avg_p:
                updated_data.append(train_data[j])

        train_data.append(predict_data)
        updated_data.append(predict_data)

        print("SELECTED:", len(updated_data))
        ###
        # UPDATE MODEL
        ###
        model = dbn.update(model, logfile.combine(updated_data))

        print("Accuracy:", metric.ACCURACY.calculate(results))


def main():
    d = data.get_data("BPIC15_1")
    m = Methods.get_prediction_method("DIMAURO")
    s = setting.STANDARD
    e = metric.CUMM_ACCURACY

    s.train_percentage = 50

    results = []

    d.prepare(s)
    d.create_batch("normal", "%Y-%m-%d %H:%M:%S")

    basic_model = m.train(d)
    res = [x for y in m.test(d) for x in y]
    store_results("results/%s_%s_normal.csv" % (m.name, d.name), res)
    accs = e.calculate(res)

    d.create_batch("day", "%Y-%m-%d %H:%M:%S")
    print("TEST DAYS")
    res_update = [x for y in m.test_and_update(d, False) for x in y]
    store_results("results/%s_%s_day.csv" % (m.name, d.name), res_update)
    accs_update_days = e.calculate(res_update)

    d.create_batch("week", "%Y-%m-%d %H:%M:%S")
    print("TEST WEEKS")
    res_update = [x for y in m.test_and_update(d, False) for x in y]
    store_results("results/%s_%s_week.csv" % (m.name, d.name), res_update)
    accs_update_weeks = e.calculate(res_update)

    d.create_batch("month", "%Y-%m-%d %H:%M:%S")
    print("TEST MONTHS")
    res_update = [x for y in m.test_and_update(d, False) for x in y]
    store_results("results/%s_%s_month.csv" % (m.name, d.name), res_update)
    accs_update_months = e.calculate(res_update)


    plt.title(d.name + " - " + m.name)
    plt.plot(range(len(accs)), accs, label="no-update")
    plt.plot(range(len(accs_update_days)), accs_update_days, label="update-retain (day)")
    plt.plot(range(len(accs_update_weeks)), accs_update_weeks, label="update-retain (week)")
    plt.plot(range(len(accs_update_months)), accs_update_months, label="update-retain (month)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    dbn_adaptive_window("BPIC15_1")

