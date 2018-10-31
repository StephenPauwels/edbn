from binet.processmining.core import Case, Event, EventLog
from binet.anomalydetection import BINetAnomalyDetector, BINetV2AnomalyDetector
from binet.dataset import Dataset

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import auc

def from_csv(file_path, test_path = None):
    """
    Load an event log from a CSV file

    :param file_path: path to CSV file
    :return: EventLog object
    """
    # parse file as pandas dataframe
    df = pd.read_csv(file_path, dtype="str")

    if test_path is not None:
        df2 = pd.read_csv(test_path, dtype="str")
        frames = [df, df2]
        df = pd.concat(frames)

    # create event log
    event_log = EventLog()

    i = 0
    # iterate by distinct trace_id
    for case_id in np.unique(df['Case']):
        _case = Case(id=case_id)
        # iterate over rows per trace_id
        for index, row in df[df.Case == case_id].iterrows():
            start_time = i
            i += 1
            activity = row['Activity']
            resource = row['Resource']
            weekday = row['Weekday']
            _event = Event(name=activity, timestamp=start_time, resource=resource, weekday=weekday)
            _case.add_event(_event)
            _case.attributes['label'] = row['Anomaly']
        event_log.add_case(_case)

    return event_log

def fit(dataset):
    anom_detector = BINetV2AnomalyDetector(embedding=True, epochs=5, batch_size=100)
    anom_detector.fit(dataset)
    return anom_detector

def fit_and_detect(dataset):
    anom_detector = BINetV2AnomalyDetector(embedding=True, epochs=10, batch_size=1000)
    anom_detector.fit(dataset)
    print("Detecting")
    anomaly_scores, predictions, attentions = anom_detector.detect(dataset)

def read_bpic_data(file):
    eventlog = from_csv(file + "BPIC15_train_1.csv_ints", file + "BPIC15_test_1.csv_ints")
    data = Dataset()
    data._event_log = eventlog
    data._load_from_event_log(eventlog)
    return data

def train_bpic():
    path = "../Data/"
    print("Read dataset")
    return fit(read_bpic_data(path))

def plot_single_roc_curve(scores, save_file=None):
    fpr, tpr = calc_roc(scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if save_file:
        plt.savefig(save_file)
    plt.show()

def calc_roc(values):
    total_pos = 0
    total_neg = 0
    for v in values:
        if not v[3]:
            total_neg += 1
        else:
            total_pos += 1
    tprs = []
    fprs = []
    true_pos = 0
    false_pos = 0
    i = 0
    for v in sorted(values, key=lambda l: l[1]):
        i += 1
        if v[3]:
            true_pos += 1
        else:
            false_pos += 1
        true_neg = 0
        false_neg = 0
        fpr = false_pos / total_neg
        tpr = true_pos / total_pos
        fprs.append(fpr)
        tprs.append(tpr)
    return fprs, tprs

def test_bpic(binet):
    path = "../Data/"
    dataset = read_bpic_data(path)
    anomaly_scores, predictions, attentions = binet.detect(dataset)

    log = dataset.event_log
    scores = []

    accum = np.mean

    for i in range(len(log.cases)):
        score = accum([e for e in anomaly_scores[i].flatten() if e > 0])
        #print(log.cases[i].attributes["label"] != "1", score)
        scores.append((log.cases[i].id, score, 0, log.cases[i].attributes["label"] != "1"))
    scores.sort(key=lambda l: l[1], reverse=True)
    for s in scores:
        print(s)
    plot_single_roc_curve(scores)


def example():
    from binet.utils import get_event_logs
    dataset = [e.name for e in get_event_logs() if e.model in ['p2p'] and e.id in [2]][0]
    data = Dataset(dataset)
    fit_and_detect(data)

if __name__ == "__main__":
    # example() # Using the example (generated) log file for both fit and detect

    binet = train_bpic() # First only train the model, looks like it works
    test_bpic(binet) # Use the model to detect anomalies on the same data as used for training


    print("Done")
