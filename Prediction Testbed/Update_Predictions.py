import matplotlib.pyplot as plt
import numpy as np

import setting
import data
import method
import metric

def store_results(file, results):
    with open(file, "w") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")

if __name__ == "__main__":
    d = data.get_data("BPIC15_1")
    m = method.get_method("DIMAURO")
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

