import pandas as pd
import matplotlib.pyplot as plt

result_file = "results.txt"

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
                  train_percentage=splitted_result[5].split(" ")[-1],
                  accuracy=splitted_result[-2].split(" ")[-1])
    records.append(record)

result_data = pd.DataFrame()
result_data = result_data.from_records(records)
result_data["accuracy"] = pd.to_numeric(result_data["accuracy"])
print(result_data.dtypes)


def calc_min_max_average(dataframe):
    min_val = dataframe["accuracy"].min()
    max_val = dataframe["accuracy"].max()
    print("Min:", min_val)
    print(dataframe[dataframe["accuracy"] == min_val].squeeze())
    print("Average:", dataframe["accuracy"].mean())
    print("Max:", dataframe["accuracy"].max())
    print(dataframe[dataframe["accuracy"] == max_val].squeeze())


def plot_acc_values(dataframe, x_axis, divide=None, title=""):
    if divide is not None:
        for group_id, group in dataframe.groupby(divide):
            plt.scatter(group[x_axis], group["accuracy"], label=group_id)
    else:
        plt.scatter(dataframe[x_axis], dataframe["accuracy"])
    plt.title(title)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.show()



data = ["Camargo_Helpdesk.csv", "Camargo_BPIC12W.csv", "Camargo_BPIC2012.csv", "BPIC15_1_sorted_new.csv",
        "BPIC15_3_sorted_new.csv", "BPIC15_5_sorted_new.csv"]
for d in data:
    print("Data:", d)
    # calc_min_max_average(result_data[result_data["data"] == d])
    plot_acc_values(result_data[result_data["data"] == d], "prefix_size", ["train_percentage"], title=d)
    print()
