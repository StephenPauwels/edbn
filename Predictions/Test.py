import multiprocessing as mp

import pandas as pd
from pgmpy.estimators import HillClimbSearch

case_attr = "Case ID"
activity_attr = "Activity"
k = 1

def create_k_context(data):
        with mp.Pool(1) as p:
        #with mp.Pool(mp.cpu_count()) as p:
            result = p.map(create_k1_context_trace, data.groupby([case_attr]))
        return pd.concat(result)


def create_k1_context_trace(trace):
    contextdata = pd.DataFrame()

    trace_data = trace[1]
    shift_data = trace_data.shift()

    joined_trace = shift_data.join(trace_data, lsuffix="_Prev0")
    joined_trace[case_attr + "_Prev0"] = trace[0]

    contextdata = contextdata.append(joined_trace, ignore_index=True)
    return contextdata

def run_pgmpy(train, test):
    est = HillClimbSearch(train)
    model = est.estimate(max_indegree=3, tabu_length=2)

    print(model.edges())

    #test_data_copy = test.copy()
    #test_data_copy.drop(activity_attr, axis=1, inplace=True)
    #test_data_copy.drop("activityNameNL", axis=1, inplace=True)
    #c_pred = model.predict(test_data_copy)
    #correct = test_data[activity_attr] == c_pred[activity_attr]

    #print(correct.value_counts())

def run_own_implementation(train, test):
    cpt_activity = {}
    for row in train.iterrows():
        activity_prev = row[1][activity_attr + "_Prev0"]
        activity = row[1][activity_attr]
        if activity_prev not in cpt_activity:
            cpt_activity[activity_prev] = {}
        if activity not in cpt_activity[activity_prev]:
            cpt_activity[activity_prev][activity] = 0
        cpt_activity[activity_prev][activity] += 1

    correct = 0
    false = 0
    for row in test.iterrows():
        activity_prev = row[1][activity_attr + "_Prev0"]
        if activity_prev in cpt_activity:
            max_count = 0
            max_activity = None
            for activity in cpt_activity[activity_prev]:
                if cpt_activity[activity_prev][activity] > max_count:
                    max_count = cpt_activity[activity_prev][activity]
                    max_activity = activity
            if row[1][activity_attr] == max_activity:
                correct += 1
            else:
                false += 1
    total = correct + false
    print("Accuracy:", correct / total)
    return correct / total


#total_values = pd.read_csv("../Data/BPIC15_5_sorted_new.csv", header=0, nrows=50000000, delimiter=",", dtype="str")
total_values = pd.read_csv("../Data/BPIC15_1_sorted_new.csv", header=0, delimiter=",", dtype="str")

k_context = create_k_context(total_values)
# Filter events where history is NaN
k_context = k_context[k_context.Activity_Prev0.notnull()]
k_context = k_context.apply(lambda x: pd.factorize(x)[0])

k_cases = list(k_context.groupby([case_attr]))

# Divide into 10 parts
"""
k_cases_chunks = np.array_split(np.array(k_cases), 10)
result = 0
for i in range(10):
    train_cases = []
    test_cases = []
    for j in range(10):
        if i != j:
            train_cases.extend(k_cases_chunks[j])
        else:
            test_cases.extend(k_cases_chunks[j])

    train_data = pd.concat([case[1] for case in train_cases])
    test_data = pd.concat([case[1] for case in test_cases])

    print("Fold:", i)
    #run_pgmpy(k_context, test_data)
    result += run_own_implementation(train_data, test_data)

print("Average result:", result / 10)
"""

train_cases = k_cases[:int(len(k_cases)*0.7)]
test_cases = k_cases[int(len(k_cases)*0.7):]

train_data = pd.concat([case[1] for case in train_cases])
test_data = pd.concat([case[1] for case in test_cases])

#print("Result", run_pgmpy(k_context, test_data))
#print("Result", run_own_implementation(k_context, test_data))
print("Result", run_own_implementation(train_data, test_data))