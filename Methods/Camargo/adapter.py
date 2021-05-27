import itertools
import math

import Methods.Camargo.embedding_training as em
import Methods.Camargo.model_training as mo
import Methods.Camargo.predict_next as pn
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

    # model_type = "specialized"
    # model_type = "concatenated"

    args = {"file_name": "data", "model_type": model_type, "norm_method": "lognorm", 'lstm_act': None,
            'l_size': 100, 'imp': 1, 'dense_act': None, 'optim': 'Nadam'}

    return mo.training_model(log, event_emb, role_emb, args, epochs, early_stop)


def update(model, log):
    vec = mo.vectorization(log)

    split = 0
    if len(log.contextdata) > 10:
        split = 0.2

    model.fit({'ac_input':vec['prefixes']['x_ac_inp'],
               'rl_input':vec['prefixes']['x_rl_inp']},
              {'act_output':vec['next_evt']['y_ac_inp'],
               'role_output':vec['next_evt']['y_rl_inp']},
              validation_split=split,
              verbose=2,
              batch_size=vec['prefixes']['x_ac_inp'].shape[1],
              epochs=10)
    return model


def test(model, log):
    return pn.predict_next(log, model)


if __name__ == "__main__":
    data = "../../Data/Helpdesk.csv"
    # data = "../../Data/BPIC15_3_sorted_new.csv"
    # data = "../../Data/BPIC12W.csv"
    case_attr = "case"
    act_attr = "event"

    logfile = LogFile(data, ",", 0, None, "completeTime", case_attr,
                      activity_attr=act_attr, convert=False, k=10)

    logfile.convert2int()

    # logfile.filter_case_length(5)
    logfile.create_k_context()
    train_log, test_log = logfile.splitTrainTest(70, case=True, method="train-test")

    model = train(train_log, epochs=200, early_stop=10)
    # model = load_model("tmp\model_rd_100 Nadam_014-5.00.h5")
    acc = test(test_log, model)
    print(acc)

