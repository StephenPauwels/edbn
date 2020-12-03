from keras.models import load_model

from RelatedMethods.Lin.Modulator import Modulator
from RelatedMethods.Lin.model import create_model, predict_next
from Utils.LogFile import LogFile


def train(log, epochs=200, early_stop=42):
    return create_model(log, "tmp", epochs, early_stop)


def test(log, model):
    return predict_next(log, model)


if __name__ == "__main__":
    # data = "../../Data/BPIC15_5_sorted_new.csv"
    data = "../../Data/BPIC12W.csv"
    case_attr = "case"
    act_attr = "event"

    logfile = LogFile(data, ",", 0, None, None, case_attr,
                      activity_attr=act_attr, convert=False, k=10)
    logfile.convert2int()
    logfile.filter_case_length(5)
    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(70, case=False, method="test-train")

    model = train(train_log, epochs=100, early_stop=10)
    # model = load_model("../../Predictions/tmp/model_001-4.51.h5", custom_objects={'Modulator': Modulator})

    acc = test(test_log, model)
    print(acc)

