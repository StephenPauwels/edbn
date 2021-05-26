import setting
import data
import method

def store_results(file, results):
    return
    with open(file, "w") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")


def store_timings(file, timings):
    with open(file, "w") as fout:
        for t in timings:
            fout.write(str(t) + "\n")

if __name__ == "__main__":
    DATASETS = ["BPIC15_1"]
    METHODS = ["TAX"]
    DRIFT = True
    RESET = [False, True]
    WINDOW = [0,1,5]
    batch = ["month"]

    DRIFT_LIST = {
        "Helpdesk": [9, 26],
        "BPIC11": [1, 9, 18],
        "BPIC12": [],
        "BPIC15_1": [2, 17, 24, 28],
        "BPIC15_2": [1, 7, 20, 27],
        "BPIC15_3": [1, 9, 15, 27],
        "BPIC15_4": [17, 20, 25],
        "BPIC15_5": [3, 20, 27]
    }

    for data_name in DATASETS:
        timeformat = "%Y-%m-%d %H:%M:%S"
        if "BPIC15" not in data_name:
            timeformat = "%Y/%m/%d %H:%M:%S.%f"


        for m in METHODS:
            d = data.get_data(data_name)
            m = method.get_method(m)
            s = setting.STANDARD
            s.train_percentage = 50

            d.prepare(s)
            d.create_batch("normal", timeformat)
            if m.name == "Di Mauro":
                m.def_params = {"early_stop": 4, "params": {"n_modules": 2}}

            import time
            start_time = time.time()
            basic_model = m.train(d.train)
            print("Runtime %s:" % m, time.time() - start_time)
            input("Waiting")
            res = m.test(basic_model, d.test_orig)

            store_results("results/%s_%s_normal.csv" % (m.name, d.name), res)

            for b in batch:
                for r in RESET:
                    if DRIFT:
                        d.create_batch(b, timeformat)
                        results, timings = m.test_and_update_drift(basic_model, d, DRIFT_LIST[data_name], r)
                        if r:
                            store_results("results/%s_%s_drift_reset.csv" % (m.name, d.name), results)
                            store_timings("results/%s_%s_drift_reset_time.csv" % (m.name, d.name), timings)
                        else:
                            store_results("results/%s_%s_drift_update.csv" % (m.name, d.name), results)
                            store_timings("results/%s_%s_drift_update_time.csv" % (m.name, d.name), timings)

                    for w in WINDOW:
                        d.create_batch(b, timeformat)
                        results, timings = m.test_and_update(basic_model, d, w, r)
                        if r:
                            store_results("results/%s_%s_%i_reset.csv" % (m.name, d.name, w), results)
                            store_timings("results/%s_%s_%i_reset_time.csv" % (m.name, d.name, w), timings)

                        else:
                            store_results("results/%s_%s_%i_update.csv" % (m.name, d.name, w), results)
                            store_timings("results/%s_%s_%i_update_time.csv" % (m.name, d.name, w), timings)


