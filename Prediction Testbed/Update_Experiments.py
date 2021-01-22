import setting
import data
import method


def store_results(file, results):
    with open(file, "w") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")


if __name__ == "__main__":
    DATASETS = data.all_data.keys()
    METHODS = ["DBN", "DIMAURO", "TAX"]
    RETAIN = [False, True]
    batch = ["day", "week", "month"]

    for d in DATASETS:
        timeformat = "%Y-%m-%d %H:%M:%S"
        if d == "Helpdesk" or d == "BPIC12":
            timeformat = "%Y/%m/%d %H:%M:%S.%f"

        for m in METHODS:
            d = data.get_data(d)
            m = method.get_method(m)
            s = setting.STANDARD
            s.train_percentage = 50

            d.prepare(s)
            d.create_batch("normal", timeformat)
            basic_model = m.train(d)

            res = [x for y in m.test(d) for x in y]
            store_results("results/%s_%s_normal.csv" % (m.name, d.name), res)

            for r in RETAIN:
                for b in batch:
                    d.create_batch(b, timeformat)
                    res_update = [x for y in m.test_and_update(d, False) for x in y]
                    if r:
                        store_results("results/%s_%s_%s_retain.csv" % (m.name, d.name, b), res_update)
                    else:
                        store_results("results/%s_%s_%s.csv" % (m.name, d.name, b), res_update)


