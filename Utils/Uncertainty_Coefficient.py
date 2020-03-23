"""
    Author: Stephen Pauwels
"""

import matplotlib.pyplot as plt
import scipy.stats as sc
import sklearn.metrics as skm


def calculate_entropy(row):
    data = row.value_counts() / len(row)
    return sc.entropy(data)


def calculate_mutual_information(col1, col2):
    return skm.mutual_info_score(col1, col2)


# Check if uncertainty coefficient between col1 and col2 is above a given threshold (ie. there exists a mapping between them)
def is_mapping(col1, col2, threshold, debug = False):
    if debug:
        print("DEBUG is mapping:", calculate_mutual_information(col1, col2) / calculate_entropy(col1))
    if calculate_entropy(col1) == 0:
        return False
    if calculate_mutual_information(col1, col2) / calculate_entropy(col1) >= threshold:
        return True
    return False


def calculate_mappings(data, attributes, threshold):
    if data.k > 0:
        filtered_data = data.get_data()[data.get_data()[attributes[0] + "_Prev0"] != 0]
    else:
        filtered_data = data.get_data()

    # Create all relations that can have a Functional Dependency
    mappings = []
    for attr1 in attributes:
        if data.isCategoricalAttribute(attr1):
            for attr2 in attributes:
                # Check attr2 -> attr1
                if data.isCategoricalAttribute(attr2) and attr1 != attr2 and is_mapping(filtered_data[attr1], filtered_data[attr2], threshold):
                    mappings.append((attr2, attr1))
                # Check attr2_PrevX -> attr1
                for i in range(data.k):
                    prev_attr2 = attr2 + "_Prev%i" % (i)
                    if prev_attr2 in data.get_data().columns:
                        debug = False
                        if is_mapping(filtered_data[attr1], filtered_data[prev_attr2], threshold, debug):
                            mappings.append((prev_attr2, attr1))
    return mappings


def calculate_new_values_rate(col):
    return len(col.unique()) / len(col)


def plot_new_values_rate(col):
    freq_new_values = []
    values = set()
    total_vals = 0
    new_vals = 0
    for v in col:
        total_vals += 1
        if v not in values:
            values.add(v)
            new_vals += 1
        freq_new_values.append(new_vals / total_vals)
    plt.plot(freq_new_values)
    plt.show()


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("../Data/BPIC15_1_sorted.csv", dtype='str')
    cols = data.columns
    candidates = []
    for col in cols:
        candidates.append([col])
    while (len(candidates) > 0):
        for cand in candidates:
            for col in cols:
                if col not in cand:
                    new_cand = cand + [col]

