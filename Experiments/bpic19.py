import pandas as pd
import eDBN.Execute as edbn
from LogFile import LogFile
import ConceptDrift as cd

train_data = LogFile("../Data/bpic2019.csv", ",", 0, 1000, "startTime", "case", "event")
model = edbn.train(train_data)

test_data = LogFile("../Data/bpic2019.csv", ",", 0, 100000, "startTime", "case", "event", values=train_data.values)
scores = cd.get_event_scores(test_data, model)
cd.plot_single_scores(scores)
cd.plot_pvalues(scores, 400)

