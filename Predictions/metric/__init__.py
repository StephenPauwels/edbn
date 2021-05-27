from Predictions.metric.accuracy import Accuracy
from Predictions.metric.cumm_accuracy import Cumm_Accuracy
from Predictions.metric.period_accuracy import Period_Accuracy
from Predictions.metric.brier import Brier
from Predictions.metric.precision import Precision
from Predictions.metric.recall import Recall

ACCURACY = Accuracy()
CUMM_ACCURACY = Cumm_Accuracy()
PERIOD_ACCURACY = Period_Accuracy(100)
BRIER = Brier()
PRECISION = Precision()
RECALL = Recall()