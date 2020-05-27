import processresults as pr
import matplotlib.pyplot as plt
import numpy as np

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

results = pr.read_result_file("split_method.txt")

DATA = ["Helpdesk.csv", "BPIC12W.csv", "BPIC12.csv", "BPIC15_1_sorted_new.csv", "BPIC15_2_sorted_new.csv",
        "BPIC15_3_sorted_new.csv", "BPIC15_4_sorted_new.csv", "BPIC15_5_sorted_new.csv"]
METHODS = ["Tax", "Taymouri", "Camargo random", "Camargo argmax", "Lin", "Di Mauro", "EDBN", "Baseline", "Pasquadibisceglie"]

DATA_NAMES = {}
DATA_NAMES["Helpdesk.csv"] = "Helpdesk"
DATA_NAMES["BPIC12W.csv"] = "BPIC12W"
DATA_NAMES["BPIC12.csv"] = "BPIC12"
DATA_NAMES["BPIC15_1_sorted_new.csv"] = "BPIC15\_1"
DATA_NAMES["BPIC15_2_sorted_new.csv"] = "BPIC15\_2"
DATA_NAMES["BPIC15_3_sorted_new.csv"] = "BPIC15\_3"
DATA_NAMES["BPIC15_4_sorted_new.csv"] = "BPIC15\_4"
DATA_NAMES["BPIC15_5_sorted_new.csv"] = "BPIC15\_5"

MARKERS = ["o", "v", "^", "s", "+", "D", "<", ">", "p"]

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]

figure = plt.figure(figsize=(25, 10))
figure.subplots_adjust(wspace=0.15, hspace=0.2)

fig_num = 241

for d in DATA:
    col_num = 0
    data_result = results[results.data == d]
    ax = figure.add_subplot(fig_num)
    for m in METHODS:
        x, y = np.array([t for t in data_result[["split_method", m]].values if t[1] != 0]).transpose()
        ax.plot(x, y, "o-", label=m, marker=MARKERS[col_num], color=tableau20[col_num])
        ax.set_ylim([-0.05, 0.9])
        col_num += 1
    ax.set_title(DATA_NAMES[d])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig_num += 1

    # plt.title(d)
    # plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)

figure.legend(labels=METHODS, loc="lower center", ncol=5, frameon=False, markerscale=2, fontsize="xx-large")
figure.savefig("splitmethod.png", bbox_inches="tight")
figure.show()