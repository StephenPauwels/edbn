import itertools
import math

import RelatedMethods.Camargo.embedding_training as em
import RelatedMethods.Camargo.model_training as mo
import RelatedMethods.Camargo.predict_next as pn
from Utils.LogFile import LogFile


def train(log, model_type="shared_cat", epochs=200, early_stop=42):
    ac_index = {val: idx + 1 for idx, val in enumerate(log.values["event"])}
    rl_index = {val: idx + 1 for idx, val in enumerate(log.values["role"])}

    # Define the number of dimensions as the 4th root of the number of categories
    dim_number = math.ceil(len(list(itertools.product(*[list(ac_index.items()),
                                                        list(rl_index.items())])))**0.25)

    ac_weights, rl_weights = em.train_embedded(log.contextdata, len(log.values["event"]) + 1,
                                               len(log.values["role"]) + 1, dim_number)

    event_emb = em.reformat_matrix(ac_weights)
    role_emb = em.reformat_matrix(rl_weights)

    args = {"file_name": "data", "model_type": model_type, "norm_method": "lognorm", 'lstm_act': None,
            'l_size': 100, 'imp': 1, 'dense_act': None, 'optim': 'Nadam'}

    return mo.training_model(log, event_emb, role_emb, args, epochs, early_stop)


def test(log, model):
    return pn.predict_next(log, model)


if __name__ == "__main__":
    data = "../../Data/Camargo_Helpdesk.csv"
    # data = "../../Data/Taymouri_bpi_12_w.csv"
    case_attr = "case"
    act_attr = "event"

    logfile = LogFile(data, ",", 0, None, None, case_attr,
                      activity_attr=act_attr, convert=False, k=4)
    logfile.convert2int()

    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(80, case=True, method="train-test")

    model = train(train_log, epochs=100, early_stop=5)
    # model = load_model("tmp\model_rd_100_Nadam_021-2.13.h5")
    acc = test(test_log, model)
    print(acc)

