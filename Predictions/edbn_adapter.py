from Utils.LogFile import LogFile
from EDBN.Execute import train as edbn_train
from EDBN.LearnBayesianStructure import Structure_learner
from Predictions.eDBN_Prediction import predict_next_event, predict_next_event_update

def train(log):
    return edbn_train(log)

def test(log, model):
    return predict_next_event(model, log)


def test_and_update(logs, model, dummy=None):
    results = []
    i = 0
    for t in logs:
        print(i, "/", len(logs))
        i += 1
        results.extend(predict_next_event_update(model, logs[t]["data"]))
    return results


def test_and_update_retain(test_logs, model, train_log):
    # Create the list of allowed edges
    restrictions = []
    attributes = list(train_log.attributes())
    for attr1 in attributes:
        if attr1 != train_log.activity:
            continue
        for attr2 in attributes:
            if attr2 not in train_log.ignoreHistoryAttributes:
                for i in range(train_log.k):
                    restrictions.append((attr2 + "_Prev%i" % i, attr1))

    learner = Structure_learner()

    results = []
    i = 0
    for t in test_logs:
        print(i, "/", len(test_logs))
        i += 1
        test_log = test_logs[t]["data"]
        results.extend(predict_next_event_update(model, test_log))

        train_log = train_log.extend_data(test_log)
        print("Length train:", train_log.contextdata.shape)

        learner.start_model(train_log, model, restrictions)
        relations = learner.learn()

        updated = False
        for relation in relations:
            if relation[0] not in [p.attr_name for p in model.get_variable(relation[1]).get_conditional_parents()]:
                model.get_variable(relation[1]).add_parent(model.get_variable(relation[0]))
                print("   ", relation[0], "->", relation[1])
                updated = True
        if updated:
            model.train(train_log)
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

