import matplotlib.pyplot as plt
import os.path
import statistics

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

colors = {"DBN": [(31, 119, 180), (174, 199, 232)],
          "SDL": [(255, 127, 14), (255, 187, 120)],
          "Tax": [(44, 160, 44), (152, 223, 138)],
          "Di Mauro": [(148, 103, 189), (197, 176, 213)]}
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

for k in colors:
    for i in range(len(colors[k])):
        r, g, b = colors[k][i]
        colors[k][i] = (r / 255., g / 255., b / 255.)

LINE_STYLE = {"DBN": "-", "SDL": "--", "Tax": "-.", "Di Mauro": ":"}

METHODS = ["DBN", "SDL", "Tax", "Di Mauro"]
DATASETS = ["Helpdesk", "BPIC11", "BPIC12", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]
RESET = [True, False]
WINDOW = [0,1,5]


def get_filename(dataset, method, reset, window):
    filename = "%s_%s_%i" % (method, dataset, window)
    if reset:
        filename += "_reset"
    else:
        filename += "_update"
    return filename


def open_file(filename):
    results = []
    with open(filename) as finn:
        for line in finn:
            splitted = line.split(",")
            results.append((int(splitted[0]), int(splitted[1]), float(splitted[2])))
    return results


def load_results_new():
    all_results = {}
    for m in METHODS:
        for d in DATASETS:
            for r in RESET:
                if r:
                    filename = "%s_%s_drift_reset" % (m, d)
                else:
                    filename = "%s_%s_drift_update" % (m,d)
                if os.path.isfile("results/" + filename + ".csv"):
                    results = open_file("results/" + filename + ".csv")
                    all_results[filename] = [1 if res[0] == res[1] else 0 for res in results]

                for w in WINDOW:
                    filename = get_filename(d, m, r, w)
                    if os.path.isfile("results/" + filename + ".csv"):
                        results = open_file("results/" + filename + ".csv")
                        all_results[filename] = [1 if res[0] == res[1] else 0 for res in results]

            filename = "%s_%s_normal" % (m, d)
            if os.path.isfile("results/" + filename + ".csv"):
                results = open_file("results/" + filename + ".csv")
                all_results[filename] = [1 if res[0] == res[1] else 0 for res in results]

    return all_results


def load_timings():
    all_timings = {}
    for m in METHODS:
        for d in DATASETS:
            for r in RESET:
                if r:
                    filename = "%s_%s_drift_reset_time" % (m, d)
                else:
                    filename = "%s_%s_drift_update_time" % (m, d)
                if os.path.isfile("results/%s.csv" % filename):
                    times = []
                    with open("results/%s.csv" % filename) as finn:
                        for line in finn:
                            times.append(float(line))
                    all_timings[filename] = times

                for w in WINDOW:
                    filename = "%s_time" % get_filename(d, m, r, w)
                    if os.path.isfile("results/%s.csv" % filename):
                        times = []
                        with open("results/%s.csv" % filename) as finn:
                            for line in finn:
                                times.append(float(line))
                        all_timings[filename] = times
    return all_timings


def result_list():
    all_results = {}
    for m in METHODS:
        for d in DATASETS:
            filename = get_filename(d, m, "normal", False)
            if os.path.isfile("results/" + filename + ".csv"):
                results = open_file("results/" + filename + ".csv")
                all_results[filename] = [r[0] == r[1] for r in results]
    return all_results


def create_latex_full_table_new(results):
    for d in DATASETS:
        line = d.replace("_", "\_") + " & No-update & "
        for m in METHODS:
            filename = "%s_%s_normal" % (m, d)
            if filename in results:
                result = accuracy(results[filename])
                line += " & %.2f" % result
            else:
                line += " & "
        line += "\\\\"
        print(line)
        # print("\cline{2-7}")
        for r in RESET:
            for w in WINDOW:
                line = ""
                if r and w == 0:
                    line += "& Reset & Full"
                elif not r and w == 0:
                    # print("\cline{2-7}")
                    line += "& Update & Full"
                else:
                    line += " & & Window (size %s)" % w

                for m in METHODS:
                    name = get_filename(d, m, r, w)
                    if name in results:
                        line += " & %.2f" % accuracy(results[name])
                    else:
                        line += " & "
                line += "\\\\"
                print(line)

            line = " & & Drift"
            for m in METHODS:
                if r:
                    name = "%s_%s_drift_reset" % (m, d)
                else:
                    name = "%s_%s_drift_update" % (m, d)
                if name in results:
                    line += " & %.2f" % accuracy(results[name])
                else:
                    line += " & "
            line += "\\\\"
            print(line)
        print("\hline")


def accuracy(y):
    return sum(y) / len(y)

def create_latex_full_table(results):
    for d in DATASETS:
        line = d.replace("_", "\_") + " & no-update & "
        for m in METHODS:
            filename = get_filename(d, m, "normal", False)
            if filename in results:
                result = results[filename][1][-1]
                line += " & %.2f" % result
            else:
                line += " & "
        line += "\\\\"
        print(line)
        print("\cline{2-7}")
        line = "& update-full & month"
        for m in METHODS:
            filename = "results/%s_%s_month_complete.csv" % (m, d)
            if filename in results:
                result = results[filename][1][-1]
                line += " & %.2f" % result
            else:
                line += " & "
        line += "\\\\"
        print(line)
        print("\cline{2-7}")
        for r in [True, False]:
            for b in ["day", "week", "month"]:
                line = ""
                if r and b == "day":
                    line += "& update-retain"
                elif not r and b == "day":
                    print("\cline{2-7}")
                    line += "& update-only"
                else:
                    line += " & "

                line += " & %s" % b
                for m in METHODS:
                    name = get_filename(d, m, b, r)
                    if name in results:
                        line += " & %.2f" % results[name][1][-1]
                    else:
                        line += " & "
                line += "\\\\"
                print(line)
        print("\hline")


def create_baseline_dataset_plot(list_results, windowsize = 1000):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]

    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(wspace=1, hspace=5)

    fig_num = 1

    for d in DATASETS:
        for m in METHODS:
            filename = "%s_%s_normal" % (m, d)
            if filename in list_results:
                result_list = list_results[filename]
                plt.subplot(5,2,fig_num)
                plt.title(d.replace("_", "\_"))
                plt.ylim(0,1)

                x = []
                y = []
                for x_i in range(windowsize, len(result_list), 1):
                    x.append(len(x))
                    y.append(sum(result_list[x_i - windowsize:x_i]) / windowsize)

                plt.plot(x, y, label=m, color=colors[m][0], linewidth=0.5)

        fig_num += 1

    plt.tight_layout()
    leg = plt.legend(loc="lower center", bbox_to_anchor=(0, -0.5), ncol=4, fontsize="xx-large")
    for line in leg.get_lines():
        line.set_linewidth(2)
    plt.savefig("dataset_overview.eps")
    plt.show()


def create_strategy_plot_new(results, dataset, windowsize = 2500):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]

    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(wspace=1, hspace=5)

    fig_num = 1
    for m in METHODS:
        i = 0
        for w in WINDOW:
            for r in RESET:
                if w == 0:
                    l = "Full - Reset" if r else "Full - Update"
                else:
                    l = "Window (l=%i) - " % w + ("Reset" if r else "Update")
                filename = get_filename(dataset, m, r, w)
                if filename in results:
                    result_list = results[filename]

                    x = []
                    y = []
                    for x_i in range(windowsize, len(result_list), 1):
                        x.append(len(x))
                        y.append(sum(result_list[x_i - windowsize:x_i]) / windowsize)

                    plt.subplot(3, 2, fig_num)
                    plt.title(m)
                    plt.ylim(0.2, 1)
                    plt.plot(x, y, color=tableau20[i], label=l, linewidth=0.5)
                i += 1

        for r in RESET:
            l = "Drift - Reset" if r else "Drift - Update"
            filename = "%s_%s_drift_%s" % (m, dataset, "reset" if r else "update")
            if filename in results:
                result_list = results[filename]

                x = []
                y = []
                for x_i in range(windowsize, len(result_list), 1):
                    x.append(len(x))
                    y.append(sum(result_list[x_i - windowsize:x_i]) / windowsize)

                plt.subplot(3, 2, fig_num)
                plt.title(m)
                plt.ylim(0.2, 1)
                plt.plot(x, y, color=tableau20[i], label=l, linewidth=0.5)
            i += 1
        fig_num += 1

    plt.tight_layout()
    leg = plt.legend(loc="lower center", bbox_to_anchor=(0, -0.5), ncol=4, fontsize="xx-large")
    for line in leg.get_lines():
        line.set_linewidth(2)
    plt.savefig("strategy_overview.eps")
    plt.show()

def create_strategy_plot(results, windowsize = 2500):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]

    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(wspace=1, hspace=5)

    fig_num = 1
    for d in DATASETS:
        for m in METHODS:
            i = 0
            for r in RESET:
                for w in WINDOW:
                    l = "%s (%i - Reset)" % (m, w) if r else "%s (%i - Update)" % (m, w)
                    filename = get_filename(d, m, r, w)
                    if filename in results:
                        result_list = results[filename]

                        x = []
                        y = []
                        for x_i in range(windowsize, len(result_list), 1):
                            x.append(len(x))
                            y.append(sum(result_list[x_i - windowsize:x_i]) / windowsize)

                        plt.subplot(5,2,fig_num)
                        plt.title(d.replace("_","\_"))
                        plt.ylim(0, 1)
                        plt.plot(x, y, label=l, ls=LINE_STYLE[m])
                    i += 1

        fig_num += 1

    plt.tight_layout()
    plt.legend(loc="lower center", bbox_to_anchor=(0, -1), ncol=4, fontsize="xx-large")
    plt.savefig("strategy_overview.eps")
    plt.show()


def create_batch_plot(results):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]

    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(wspace=1, hspace=5)

    fig_num = 321
    for d in ["BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        col_idx = 0

        for m in METHODS:
            for b in ["day", "week", "month"]:
                filename = get_filename(d, m, b, False)
                if filename in results:
                    x, y = results[filename]
                    plt.subplot(fig_num)
                    plt.title(d.replace("_","\_"))
                    plt.ylim(0, 1)
                    plt.plot(x, y, label=("%s (%s)" % (m, b)), color=tableau20[col_idx], ls=LINE_STYLE[m])
                    col_idx += 1

        fig_num += 1

    plt.tight_layout()
    plt.legend(loc="lower right", bbox_to_anchor=(2, 0.1), ncol=2, fontsize="xx-large")
    plt.savefig("batch_overview.eps")
    plt.show()


def create_compare_normal_plot(results, windowsize = 1000):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]

    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(wspace=1, hspace=5)

    fig_num = 1
    for d in DATASETS:
        for m in METHODS:
            filename = "%s_%s_normal" % (m, d)
            if filename in results:
                result_list = results[filename]

                x = []
                y = []
                for x_i in range(windowsize, len(result_list), 1):
                    x.append(len(x))
                    y.append(sum(result_list[x_i - windowsize:x_i]) / windowsize)

                plt.subplot(5, 2, fig_num)
                plt.title(d.replace("_","\_"))
                plt.ylim(0, 1)
                plt.plot(x, y, label="%s (No-update)" % m, color=colors[m][0], linewidth=0.5)

            filename = get_filename(d, m, False, 1)
            if filename in results:
                result_list = results[filename]

                x = []
                y = []
                for x_i in range(windowsize, len(result_list), 1):
                    x.append(len(x))
                    y.append(sum(result_list[x_i - windowsize:x_i]) / windowsize)

                plt.subplot(5, 2, fig_num)
                plt.title(d.replace("_","\_"))
                plt.ylim(0, 1)
                plt.plot(x, y, label="%s (Update W=1)" % m, color=colors[m][1], linewidth=0.5)

        fig_num += 1

    plt.tight_layout()
    leg = plt.legend(loc="lower center", bbox_to_anchor=(0, -0.7), ncol=4, fontsize="xx-large")
    for line in leg.get_lines():
        line.set_linewidth(2)
    plt.savefig("normal_compare.eps")
    plt.show()


def create_timing_table(timings, dataset):
    for m in METHODS:
        for w in WINDOW:
            for r in [True, False]:
                filename = "%s_time" % get_filename(dataset, m, r, w)
                if filename in timings:
                    print(m,w,r, statistics.mean(timings[filename]), statistics.stdev(timings[filename]))
                else:
                    print(m,w,r, "No results")

        for r in [True, False]:
            filename = "%s_%s_drift_%s_time" % (m, dataset, "reset" if r else "update")
            if filename in timings:
                print(m, "drift", r, statistics.mean(timings[filename]), statistics.stdev(timings[filename]))
            else:
                print(m, "drift", r, "No results")


if __name__ == "__main__":
    # results = load_results_new()
    timings = load_timings()
    create_timing_table(timings, "BPIC15_1")
    # list_results = result_list()
    # create_baseline_dataset_plot(results)
    # create_strategy_plot_new(results, "BPIC15_1")
    # create_batch_plot(results)
    # create_compare_normal_plot(results)
    # create_latex_full_table_new(results)
    # create_cold_plot(results)
