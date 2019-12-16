from pgmpy.models import BayesianModel
from pgmpy.estimators.StructureScore import StructureScore
from pgmpy.estimators import K2Score
from pgmpy.estimators import BicScore

from Utils.LogFile import LogFile

used_score = K2Score

logfile = LogFile("../Data/BPIC12.csv", ",", 0, 50000000, "completeTime", "case", activity_attr="task")
logfile.create_k_context()

model = BayesianModel()
print(used_score(logfile.contextdata).score(model))

model.add_edge("task_Prev0", "task")
print(used_score(logfile.contextdata).score(model))
model.add_edge("task_Prev1", "task")
print(used_score(logfile.contextdata).score(model))


model = BayesianModel()
model.add_edge("task_Prev1", "task")
print(used_score(logfile.contextdata).score(model))
model.add_edge("task_Prev0", "task")
print(used_score(logfile.contextdata).score(model))