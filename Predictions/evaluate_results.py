import csv

# TODO: make same evaluation about results from our method
# TODO: Do we also always find the most common trace?

camargo_correct_dict = {}
camargo_wrong_dict = {}

with open("../Camargo/output_files/output_run3b/HELPDESK_20000000_2_1_0_1_1/shared_cat/model_rd_100_Nadam_44-0.31_rep_0_next.csv") as finn:
    reader = csv.reader(finn, delimiter=",")
    next(reader)
    for row in reader:
        prefix = row[0]
        next_act = int(row[1])
        correct = row[5]
        prediction = (prefix, next_act)
        if correct == "1":
            camargo_correct_dict[prediction] = camargo_correct_dict.get(prediction, 0) + 1
        else:
            camargo_wrong_dict[prediction] = camargo_wrong_dict.get(prediction, 0) + 1

print("Correct:", len(camargo_correct_dict))
print("Wrong:", len(camargo_wrong_dict))

keys = sorted(camargo_wrong_dict.keys(), key=lambda l: camargo_wrong_dict[l], reverse=True)

for k in keys:
    print(k, camargo_wrong_dict[k])

doubles = 0
for k in keys:
    if k in camargo_wrong_dict:
        doubles += 1
print(doubles)

edbn_correct_dict = {}
edbn_wrong_dict = {}

with open("results_next_event.csv") as finn:
    reader = csv.reader(finn, delimiter=",")
    for row in reader:
        trace = row[0]
        correct = row[1]
        if correct == "1":
            edbn_correct_dict[trace] = edbn_correct_dict.get(trace, 0) + 1
        else:
            edbn_wrong_dict[trace] = edbn_wrong_dict.get(trace, 0) + 1

keys = sorted(edbn_wrong_dict.keys(), key=lambda l: edbn_wrong_dict[l], reverse=True)

print("Correct:", len(edbn_correct_dict))
print("Wrong:", len(edbn_wrong_dict))

for k in keys:
    print(k, edbn_wrong_dict[k])
