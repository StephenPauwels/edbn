from ICPM2020 import processresults as pr

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
METHODS = ["Tax", "Camargo argmax", "Lin", "Di Mauro", "Pasquadibisceglie", "Taymouri", "EDBN", "Baseline", "New"]
SETTINGS = ["Tax", "Camargo", "Lin", "Di Mauro", "Pasquadibisceglie", "Taymouri", "Baseline"]

scores = {}

for setting in SETTINGS:
    results = pr.read_result_file("paper_%s.txt" % (setting.lower().replace(" ", "")))
    scores[setting] = results[METHODS].mean().sort_values()

LATEX_ROW = {}

for setting in SETTINGS:
    score = scores[setting]
    i = 9
    for score_item in score.iteritems():
        print(score_item)
        if score_item[0] not in LATEX_ROW:
            LATEX_ROW[score_item[0]] = ""
        if i == 1:
            LATEX_ROW[score_item[0]] += "& \\textbf{%i \\emph{(%.2f)}} " % (i, score_item[1])
        else:
            LATEX_ROW[score_item[0]] += "& %i \\emph{(%.2f)} " % (i, score_item[1])
        i -= 1

for method in METHODS:
    print(method, LATEX_ROW[method], "\\\\")





