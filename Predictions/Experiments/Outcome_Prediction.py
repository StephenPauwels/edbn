from Utils.LogFile import LogFile



def run_edbn():
    from eDBN_Prediction import get_probabilities
    from Methods.EDBN.Train import train

    labeled_logfile = "../Data/Outcome_Prediction/BPIC15_1_f2.csv"

    log = LogFile(labeled_logfile, ";", 0, None, "time_timestamp", "Case_ID",
                  activity_attr="label", convert=True, k=1)

    columns = ["label", "Case_ID", "time_timestamp", "Activity", "monitoringResource", "question", "org_resource", "Responsible_actor", "SUMleges"]
    log.keep_attributes(columns)

    log.create_k_context()

    train_log, test_log = log.splitTrainTest(80, True, "train-test")

    train_log.ignoreHistoryAttributes.add("label")

    model = train(train_log)

    results1 = []
    results2 = []

    for case in test_log.get_cases():
        case_df = case[1]
        case_probs = {1: 1, 2: 1}
        ground = 0
        for row in case_df.iterrows():
            ground = getattr(row[1], "label")

            parents = model.variables["label"].conditional_table.parents

            value = []
            for parent in parents:
                value.append(getattr(row[1], parent.attr_name))
            tuple_val = tuple(value)

            activity_var = model.variables["label"]
            probs, unknown = get_probabilities(activity_var, tuple_val, parents)
            case_probs[1] += probs.get(1, 0)
            case_probs[2] += probs.get(2, 0)

        # correct_prob = sum(case_probs) / len(case_probs)
        if ground == 1:
            if case_probs[1] > case_probs[2]:
                results1.append(1)
            else:
                results1.append(0)

        if ground == 2:
            if case_probs[2] > case_probs[1]:
                results2.append(1)
            else:
                results2.append(0)


    print(len(results1), sum(results1) / len(results1))
    print(len(results2), sum(results2) / len(results2))


def run_sdl():
    from Methods.SDL.sdl import train, test

    labeled_logfile = "../Data/Outcome_Prediction/BPIC15_1_f2.csv"

    log = LogFile(labeled_logfile, ";", 0, None, "time_timestamp", "Case_ID",
                  activity_attr="label", convert=True, k=10)

    columns = ["label", "Case_ID", "Activity", "monitoringResource", "question", "org_resource", "Responsible_actor", "SUMleges"]
    log.keep_attributes(columns)

    log.create_k_context()

    train_log, test_log = log.splitTrainTest(80, True, "train-test")

    train_log.ignoreHistoryAttributes.add("label")
    test_log.ignoreHistoryAttributes.add("label")

    model = train(train_log, 200, 42)

    print(test(test_log, model))

    results1 = []
    results2 = []

    for case in test_log.get_cases():
        pass


if __name__ == "__main__":
    run_sdl()