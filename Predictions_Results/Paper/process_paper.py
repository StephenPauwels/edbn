import processresults as pr

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

DATA = ["Helpdesk.csv", "BPIC12W.csv", "BPIC12.csv", "BPIC15_1_sorted_new.csv", "BPIC15_2_sorted_new.csv",
        "BPIC15_3_sorted_new.csv", "BPIC15_4_sorted_new.csv", "BPIC15_5_sorted_new.csv"]
# METHODS = ["Tax", "Taymouri", "Camargo random", "Camargo argmax", "Lin", "Di Mauro", "EDBN", "Baseline", "Pasquadibisceglie"]
# METHODS = ["Tax", "Camargo argmax", "Lin", "Di Mauro", "Pasquadibisceglie", "Taymouri", "EDBN", "Baseline", "New", "EDBN_update"]
METHODS = ["Tax", "Camargo argmax", "Lin", "Di Mauro", "Pasquadibisceglie", "Taymouri", "EDBN", "Baseline"]
SETTINGS = ["Tax", "Camargo", "Lin", "Di Mauro", "Pasquadibisceglie", "Taymouri", "Baseline"]

scores = {}
scores_detail = {}

for setting in SETTINGS:
    results = pr.read_result_file("paper_%s.txt" % (setting.lower().replace(" ", "")))
    scores[setting] = results[METHODS].mean().sort_values()
    scores_detail[setting] = results

LATEX_ROW = {}

for setting in SETTINGS:
    score = scores[setting]
    i = 8
    for score_item in score.iteritems():
        print(score_item)
        if score_item[0] not in LATEX_ROW:
            LATEX_ROW[score_item[0]] = ""
        if i == 1:
            LATEX_ROW[score_item[0]] += "& \\textbf{%i \\emph{(%.2f)}} " % (i, score_item[1])
        else:
            LATEX_ROW[score_item[0]] += "& %i \\emph{(%.2f)} " % (i, score_item[1])
        i -= 1

print("OVERALL AVERAGE")
for method in METHODS:
    print(method, LATEX_ROW[method], "\\\\")


for key in scores_detail:
    print(key)
    LATEX_ROW = {}

    cols = ["data"]
    cols.extend(METHODS)
    score = scores_detail[key][cols]
    print("\\begin{table*}")
    print("\\scriptsize")
    print("\\begin{tabular}{c | c c c c c c c c}")

    print(" & ".join(cols), end="\\\\\n")
    print("\hline")
    for score_item in score.iterrows():
        print(score_item[1][0].replace("_", "\_"), "&", " & ".join([ "%.2f" % s if s != max(score_item[1][1:]) else "\\textbf{%.2f}" % s for s in list(score_item[1][1:])]), end="\\\\\n")
    print("\\end{tabular}")
    print("\\end{table*}")


