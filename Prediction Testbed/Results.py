import matplotlib.pyplot as plt
import os.path

tableau20 = [(148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

colors = {"DBN": [(31, 119, 180), (174, 199, 232)],
          "SDL": [(255, 127, 14), (255, 187, 120)],
          "Tax": [(44, 160, 44), (152, 223, 138)],
          "Di Mauro": [(214, 39, 40), (255, 152, 150)]}
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
WINDOW = [0,1]


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
            for w in WINDOW:
                for r in RESET:
                    filename = get_filename(d, m, r, w)
                    if os.path.isfile("results/" + filename + ".csv"):
                        results = open_file("results/" + filename + ".csv")
                        all_results[filename] = get_accuracy(results)

            filename = "%s_%s_normal" % (m, d)
            if os.path.isfile("results/" + filename + ".csv"):
                results = open_file("results/" + filename + ".csv")
                all_results[filename] = get_accuracy(results)

    return all_results

def get_accuracy(results):
    x = []
    y = []
    total = 0
    correct = 0
    for res in results:
        x.append(total)
        if res[0] == res[1]:
            correct += 1
        total += 1
        y.append(correct / total)
    return y

def load_results(cold=False):
    all_results = {}
    for m in METHODS:
        for d in DATASETS:
            complete_filename = "results/%s_%s_month_complete.csv" % (m, d)
            if os.path.isfile(complete_filename):
                results = open_file(complete_filename)

                x = []
                y = []
                total = 0
                correct = 0
                for res in results:
                    x.append(total)
                    if res[0] == res[1]:
                        correct += 1
                    total += 1
                    y.append(correct / total)
                all_results[complete_filename] = (x, y)

            for b in BATCH:
                for r in RETAIN:
                    if m == "DBN" and cold:
                        filename = get_filename(d, m, b, False)
                    else:
                        filename = get_filename(d, m, b, r, r)
                    if cold:
                        filename = "Cold_" + filename
                    if os.path.isfile("results/" + filename + ".csv"):
                        results = open_file("results/" + filename + ".csv")

                        x = []
                        y = []
                        total = 0
                        correct = 0
                        for res in results:
                            x.append(total)
                            if res[0] == res[1]:
                                correct += 1
                            total += 1
                            y.append(correct / total)

                        all_results[filename] = (x,y)
    return all_results


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
                result = results[filename][-1]
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
                        line += " & %.2f" % results[name][-1]
                    else:
                        line += " & "
                line += "\\\\"
                print(line)
        print("\hline")


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


def create_baseline_dataset_plot(list_results, windowsize = 500):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]

    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(wspace=1, hspace=5)

    fig_num = 1

    for d in DATASETS:
        for m in METHODS:
            filename = get_filename(d, m, "normal", False)
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


                plt.plot(x, y, label=m, color=colors[m][0], ls=LINE_STYLE[m])

        fig_num += 1

    plt.tight_layout()
    plt.legend(loc="lower center", bbox_to_anchor=(0, -0.5), ncol=4, fontsize="xx-large")
    plt.savefig("dataset_overview.eps")
    plt.show()


def create_strategy_plot(results):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]

    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(wspace=1, hspace=5)

    fig_num = 321
    for d in ["BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        col_idx = 0

        for m in METHODS:
            for r in [False, True]:
                strat = ""
                if r:
                    strat = "update-retain"
                else:
                    strat = "update-only"
                filename = get_filename(d, m, "day", r)
                if filename in results:
                    x, y = results[filename]
                    plt.subplot(fig_num)
                    plt.title(d.replace("_","\_"))
                    plt.ylim(0, 1)
                    plt.plot(x, y, label=("%s (%s)" % (m, strat)), color=tableau20[col_idx], ls=LINE_STYLE[m])
                    col_idx += 1

        fig_num += 1

    plt.tight_layout()
    plt.legend(loc="lower right", bbox_to_anchor=(2, 0.5), ncol=2, fontsize="xx-large")
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


def create_compare_normal_plot(results):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]

    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(wspace=1, hspace=5)

    fig_num = 321
    for d in ["BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        col_idx = 0

        for m in METHODS:
            for b in ["day", "normal"]:
                for r in [False, True]:
                    filename = get_filename(d, m, b, r)
                    if filename in results:
                        x, y = results[filename]
                        plt.subplot(fig_num)
                        plt.title(d.replace("_","\_"))
                        plt.ylim(0, 1)
                        if b == "normal":
                            plt.plot(x, y, label=("%s (%s)" % (m, "no-update")), color=tableau20[col_idx], ls=LINE_STYLE[m])
                        elif b == "day" and not r:
                            plt.plot(x, y, label=("%s (%s)" % (m, "update-only")), color=tableau20[col_idx], ls=LINE_STYLE[m])
                        elif b == "day" and r:
                            plt.plot(x, y, label=("%s (%s)" % (m, "update-retain")), color=tableau20[col_idx], ls=LINE_STYLE[m])
                        col_idx += 1

        fig_num += 1

    plt.tight_layout()
    plt.legend(loc="lower right", bbox_to_anchor=(2, 0.1), ncol=2, fontsize="xx-large")
    plt.savefig("normal_compare.eps")
    plt.show()


def create_cold_plot(results):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]

    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(wspace=1, hspace=5)

    fig_num = 421
    for d in ["Helpdesk", "BPIC12", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
        col_idx = 0

        for m in METHODS:
            if m == "DBN":
                filename = get_filename(d, m, "month", True)
            else:
                filename = get_filename(d, m, "month", False, False)
            filename = "Cold_" + filename
            print(filename)
            if filename in results:
                x, y = results[filename]
                plt.subplot(fig_num)
                plt.title(d.replace("_","\_"))
                plt.ylim(0, 1)
                plt.plot(x, y, label=("%s" % m), color=tableau20[col_idx], ls=LINE_STYLE[m])
                col_idx += 1
            else:
                print("Not found")
        fig_num += 1

    plt.tight_layout()
    plt.legend(loc="lower right", bbox_to_anchor=(2, 0.1), ncol=2, fontsize="xx-large")
    plt.savefig("cold_start.eps")
    plt.show()


if __name__ == "__main__":
    results = load_results_new()
    # list_results = result_list()
    # create_baseline_dataset_plot(list_results)
    # create_strategy_plot(results)
    # create_batch_plot(results)
    # create_compare_normal_plot(results)
    create_latex_full_table_new(results)
    # create_cold_plot(results)
