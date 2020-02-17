import EDBN.Execute as edbn
from Utils.LogFile import LogFile

train = "../Data/BPIC15_1_sorted_new.csv"
test = "../Data/BPIC15_1_sorted_new.csv"

train_data = LogFile(train, ",", 0, 20000, "Complete Timestamp", "Case ID", activity_attr="Activity", convert=False)
train_data.remove_attributes(["Anomaly", "Type", "Time"])
train_data.convert2int()

edbn_model = edbn.train(train_data)

test_data = LogFile(test, ",", 0, 5000000, "Complete Timestamp", "Case ID", activity_attr="Activity", values=train_data.values)
test_data.remove_attributes(["Anomaly", "Type", "Time"])
test_data.create_k_context()
context = test_data.get_data()
print(context)

activity_variable = edbn_model.variables["Activity"]
activity_fdt = activity_variable.fdt
fdt_parents = activity_variable.functional_parents
activity_cpt = activity_variable.cpt
cpt_parents = activity_variable.conditional_parents

print("FDT parents:", fdt_parents)
print("CPT parents:", cpt_parents)

correct = 0
false = 0
for position in range(1,26485):
    row = context.loc[position]
    if row["Activity_Prev0"] == 0:
        continue

    min_prob = 0
    pred_val = None

    candidates = []
    # Filter according to CPT
    if len(cpt_parents) == 1:
        print("Testing")
        if row[cpt_parents[0].attr_name] in activity_cpt:
            cpt_row = activity_cpt[row[cpt_parents[0].attr_name]]
            for value in cpt_row:
                if cpt_row[value] > min_prob:
                    pred_val = value
                    min_prob = cpt_row[value]

    # TODO Select candidate from Functional Dependencies
    candidates = []
    for i in range(len(activity_variable.functional_parents)):
        pass

    if pred_val is None or pred_val != row["Activity"]:
        false += 1
    else:
        correct += 1
total = false + correct
print("Correct:", correct, "(" + str(correct / total * 100) + "%)")
print("False:", false, "(" + str(false / total * 100) + "%)")


