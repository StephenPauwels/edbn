import pandas as pd

import Uncertainty_Coefficient as uc


data = pd.read_csv("../Data/Experiments/0_train_0.csv_ints", delimiter=",", nrows=100000000, header=0, dtype=int, skiprows=0)

for col1 in data:
    if col1 != "Anomaly":
        for col2 in data:
            print(col1, col2, uc.calculate_mutual_information(data[col1], data[col2]) / uc.calculate_entropy(data[col1]))