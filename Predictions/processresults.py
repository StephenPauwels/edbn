import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["Tax", "Taymouri", "Camargo_random", "Camargo_argmax", "Lin", "Di Mauro", "EDBN", "Baseline"]
DATA = ["Camargo_Helpdesk.csv", "Camargo_BPIC12W.csv", "Camargo_BPIC2012.csv", "BPIC15_1_sorted_new.csv",
        "BPIC15_3_sorted_new.csv", "BPIC15_5_sorted_new.csv"]

result_file = "test_end_event.txt"

with open(result_file, "r") as finn:
    result_string = finn.read()

single_results = result_string.split("====================================")[:-1]

records = []
for result in single_results:
    splitted_result = result.split("\n")
    if splitted_result[0] == "":
        splitted_result = splitted_result[2:]
    record = dict(data=splitted_result[0].split(" ")[-1].split("/")[-1],
                  prefix_size=splitted_result[1].split(" ")[-1],
                  end_event=splitted_result[2].split(" ")[-1],
                  split_method=splitted_result[3].split(" ")[-1],
                  split_cases=splitted_result[4].split(" ")[-1],
                  train_percentage=splitted_result[5].split(" ")[-1])
    for acc_string in splitted_result[8:-1]:
        method, acc = acc_string.split(": ")
        if method == "Camargo":
            acc_list = eval(acc)
            record["Camargo_random"] = float(acc_list[0])
            record["Camargo_argmax"] = float(acc_list[1])
        else:
            record[method] = float(acc)
    records.append(record)

result_data = pd.DataFrame()
result_data = result_data.from_records(records)
print(result_data.dtypes)


def plot_acc_values(dataframe, method, x_axis, divide=None, title=""):
    if divide is not None:
        for group_id, group in dataframe.groupby(divide):
            plt.scatter(group[x_axis], group[method], label=group_id)
    else:
        plt.scatter(dataframe[x_axis], dataframe[method])
    plt.title(title)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.show()


def plot_all_acc_values(dataframe, x_axis, title=""):
    for m in METHODS:
        plt.plot(dataframe[x_axis], dataframe[m], "-o", label=m)
    plt.title(title)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.show()


def get_max_per_x(dataframe, x_axis):
    # dataframe = dataframe[dataframe["end_event"] == "False"]
    result = {}
    for prefix, group in dataframe.groupby(x_axis):
        for m in METHODS:
            if m not in result:
                result[m] = []
            result[m].append((prefix, max(group[m])))
    return result


def plot_max_acc_values(max_vals, title=""):
    for method in max_vals:
        x = [i[0] for i in max_vals[method]]
        y = [i[1] for i in max_vals[method]]
        plt.plot(x, y, "-o", label=method)
    plt.ylim([0.7, 1])
    plt.title(title)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.show()


# for d in DATA:
#     plot_acc_values(result_data[result_data["data"] == d], "Baseline", "prefix_size", "end_event", title=d)
# plot_max_acc_values(get_max_per_x(result_data[result_data["data"] == DATA[0]], "prefix_size"), "Helpdesk - Max per method")
plot_all_acc_values(result_data[result_data["data"] == DATA[0]], "end_event", title=DATA[0])
