from metric.accuracy import Accuracy
from metric.cumm_accuracy import Cumm_Accuracy
from metric.period_accuracy import Period_Accuracy
from metric.brier import Brier
from metric.precision import Precision
from metric.recall import Recall

ACCURACY = Accuracy()
CUMM_ACCURACY = Cumm_Accuracy()
PERIOD_ACCURACY = Period_Accuracy(100)
BRIER = Brier()
PRECISION = Precision()
RECALL = Recall()