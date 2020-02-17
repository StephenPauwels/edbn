import Preprocessing as data

dataset = data.BPIC12W
dataset_size = 500000
k = 1
add_end = True
reduce_tasks = False
resource_pools = False
remove_resource = False

logfile = data.get_data(dataset, dataset_size, k, add_end, reduce_tasks, resource_pools, remove_resource)
logfile.create_k_context()

train_log, test_log = logfile.splitTrainTest(70)

END_EVENT = train_log.convert_string2int(train_log.activity, "END")

activity_tree = {"START": {}, "COUNT": 0}
number_of_leafs = 0
max_trace_length = 0


for case_name, case in train_log.get_data().groupby([train_log.trace]):
    trace = case[train_log.activity].values
    if len(trace) > max_trace_length:
        max_trace_length = len(trace)
    current_node = activity_tree["START"]
    activity_tree["COUNT"] += 1
    for event in trace:
        if event not in current_node and event != END_EVENT:
            current_node[event] = {"COUNT": 1}
            current_node = current_node[event]
        elif event not in current_node and event == END_EVENT:
            current_node[event] = {"COUNT": 1}
            number_of_leafs += 1
        elif event in current_node and event == END_EVENT:
            current_node[event]["COUNT"] += 1
        else:
            current_node[event]["COUNT"] += 1
            current_node = current_node[event]

print("ACTIVITY TREE BUILT")
print(activity_tree)
print("Branches:", number_of_leafs)
print("Max trace length:", max_trace_length)

match = 0
no_match = 0

for case_name, case in test_log.get_data().groupby([test_log.trace]):
    trace = case[test_log.activity].values
    current_node = activity_tree["START"]
    not_found = False
    for event in trace:
        if event == END_EVENT:
            match += 1
            break
        if event not in current_node:
            no_match += 1
            break
        else:
            current_node = current_node[event]
print(match, no_match)
print(match / (match + no_match))
print()
print()
print("PREDICTING NEXT EVENT")
correct = 0
not_correct = 0
for case_name, case in test_log.get_data().groupby([test_log.trace]):
    trace = case[test_log.activity].values
    current_node = activity_tree["START"]
    if trace[0] not in current_node:
        not_correct += 1
    else:
        current_node = current_node[trace[0]] # Assuming the first event is always know
    for i in range(1, len(trace)):
        max_count = 0
        max_val = None
        for k in current_node:
            if k != "COUNT":
                if current_node[k]["COUNT"] > max_count:
                    max_count = current_node[k]["COUNT"]
                    max_val = k
        if trace[i] == max_val:
            correct += 1
            current_node = current_node[trace[i]]
        else:
            if trace[i] in current_node:
                current_node = current_node[trace[i]]
            not_correct += 1
print(correct, not_correct)
print(correct / (correct + not_correct))
