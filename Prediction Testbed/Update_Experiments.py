import setting
import data
import method

def store_results(file, results):
    with open(file, "w") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")


if __name__ == "__main__":
    DATASETS = ["BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]#["Helpdesk", "BPIC12", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]
    METHODS = ["SDL", "DBN", "TAX", "DIMAURO"]
    RETAIN = []
    COMPLETE_RETRAIN = True
    batch = ["day", "week", "month"]

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
            basic_model = m.train(d)

            res = [x for y in m.test(d) for x in y]
            store_results("results/%s_%s_normal.csv" % (m.name, d.name), res)

            if COMPLETE_RETRAIN:
                d.create_batch("month", timeformat)
                res_complete = m.test_and_full_update(d)
                store_results("results/%s_%s_month_complete.csv" % (m.name, d.name), res_complete)

            for r in RETAIN:
                for b in batch:
                    d.create_batch(b, timeformat)
                    res_update = [x for y in m.test_and_update(d, r) for x in y]
                    if r:
                        store_results("results/%s_%s_%s_retain.csv" % (m.name, d.name, b), res_update)
                    else:
                        store_results("results/%s_%s_%s.csv" % (m.name, d.name, b), res_update)


