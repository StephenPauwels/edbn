import setting
import data
import method

def store_results(file, results):
    with open(file, "w") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")


if __name__ == "__main__":
    DATASETS = ["BPIC11", "BPIC12", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]
    METHODS = ["DIMAURO"] #, "DBN", "DIMAURO", "TAX"]
    RESET = [True]
    WINDOW = [1]
    batch = ["month"]

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
            basic_model = m.train(d.train)

            res = m.test(basic_model, d.test_orig)

            store_results("results/%s_%s_normal.csv" % (m.name, d.name), res)

            for b in batch:
                for r in RESET:
                    for w in WINDOW:
                        d.create_batch(b, timeformat)
                        results = m.test_and_update(basic_model, d, w, r)
                        if r:
                            store_results("results/%s_%s_%i_reset.csv" % (m.name, d.name, w), results)
                        else:
                            store_results("results/%s_%s_%i_update.csv" % (m.name, d.name, w), results)


