from Utils.LogFile import LogFile
from EDBN.Execute import train as edbn_train
from Predictions.eDBN_Prediction import predict_next_event

def train(log):
    return edbn_train(log)

def test(log, model):
    return predict_next_event(model, log)


if __name__ == "__main__":
    data = "../Data/Helpdesk.csv"
    # data = "../../Data/Taymouri_bpi_12_w.csv"
    case_attr = "case"
    act_attr = "event"

    logfile = LogFile(data, ",", 0, None, None, case_attr,
                      activity_attr=act_attr, convert=False, k=1)
    logfile.keep_attributes(["case", "event", "role"])
    logfile.convert2int()

    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(80, case=True, method="random")

    model = train(train_log)
    acc = test(test_log, model)
    print(acc)

