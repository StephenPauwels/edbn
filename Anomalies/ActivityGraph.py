import pandas as pd

# Preprocess input file from csv format with multiple attributes to csv format with every activity as attribute
data_input = "../Data/BPIC15_1_sorted.csv"
log = pd.read_csv(data_input, header=0, dtype="str")[["Case ID", "Activity"]]

activity_names = log["Activity"].unique()

cols = activity_names + ["Case"]
converted_log = pd.DataFrame(columns=cols)

for index, row in log.iterrows():
    append_df = pd.DataFrame({row["Activity"]: [1], "Case:": [row["Case ID"]]})
    converted_log = converted_log.append(append_df, ignore_index=True, sort=False)



converted_log = converted_log.fillna(0).astype("int")
print(converted_log)
