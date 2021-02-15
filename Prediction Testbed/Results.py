import matplotlib.pyplot as plt
import os.path

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

LINE_STYLE = {"DBN": "-", "SDL": "--", "Tax": "-.", "Di Mauro": ":"}

METHODS = ["DBN", "SDL", "Tax", "Di Mauro"]
DATASETS = ["Helpdesk", "BPIC12", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5", "BPIC17", "BPIC19", "BPIC11", "SEPSIS"]
BATCH = ["day", "week", "month", "normal"]
RETAIN = [False, True]


def get_filename(dataset, method, batch, retain, multi=False):
    filename = "%s_%s_%s" % (method, dataset, batch)
    if retain:
        return filename + "_retain"
    else:
        if method != "DBN" and multi:
            return filename + "_multi"
        else:
            return filename


def open_file(filename):
    results = []
    with open(filename) as finn:
        for line in finn:
            splitted = line.split(",")
            results.append((int(splitted[0]), int(splitted[1]), float(splitted[2])))
    return results


def load_results(cold=False):
    all_results = {}
    for m in METHODS:
        for d in DATASETS:
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


def create_baseline_dataset_plot(results):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]

    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(wspace=1, hspace=5)

    fig_num = 1

    for d in DATASETS:
        col_idx = 0
        for m in METHODS:
            filename = get_filename(d, m, "normal", False)
            if filename in results:
                x,y = results[filename]
                plt.subplot(6,2,fig_num)
                plt.title(d.replace("_", "\_"))
                plt.ylim(0,1)
                plt.plot(x, y, label=m, color=tableau20[col_idx], ls=LINE_STYLE[m])
                col_idx += 1

        fig_num += 1
        
    plt.tight_layout()
    plt.legend(loc="lower right", bbox_to_anchor=(2, 0.5), ncol=4, fontsize="xx-large")
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
    results = load_results()
    # create_baseline_dataset_plot(results)
    # create_strategy_plot(results)
    # create_batch_plot(results)
    # create_compare_normal_plot(results)
    create_latex_full_table(results)
    # create_cold_plot(results)
