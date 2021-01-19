from Utils.LogFile import LogFile
from EDBN.Execute import train as edbn_train
from Predictions.eDBN_Prediction import predict_next_event, predict_next_event_update

def train(log):
    return edbn_train(log)

def test(log, model):
    return predict_next_event(model, log)


def test_and_update(logs, model, dummy=None):
    results = []
    for t in logs:
        results.extend(predict_next_event_update(model, logs[t]["data"]))
    return results


if __name__ == "__main__":
    # data = "../Data/Helpdesk.csv"
    # data = "../../Data/Taymouri_bpi_12_w.csv"
    data = "../Data/BPIC12W.csv"
    case_attr = "case"
    act_attr = "event"

    logfile = LogFile(data, ",", 0, None, None, case_attr,
                      activity_attr=act_attr, convert=False, k=4)
    logfile.keep_attributes(["case", "event", "role"])
    logfile.convert2int()
    # logfile.filter_case_length(5)

    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(70, case=True, method="test-train")

    model = train(train_log)
    acc = test(test_log, model)
    print(acc)

    import base_adapter
    model2 = base_adapter.train(train_log, 100, 10)
    acc2 = base_adapter.test(test_log, model2)
    print(acc2)

