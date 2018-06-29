import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sc
import sklearn.metrics as skm


def calculate_entropy(row):
    data = row.value_counts() / len(row)
    return sc.entropy(data)

def calculate_mutual_information(col1, col2):
    return skm.mutual_info_score(col1, col2)

# Check if uncertainty coefficient between col1 and col2 is above a given threshold (ie. there exists a mapping between them)
def is_mapping(col1, col2, threshold):
    if calculate_entropy(col1) == 0:
        return False
    if calculate_mutual_information(col1, col2) / calculate_entropy(col1) > threshold:
        return True
    return False

def calculate_mappings(data, attributes, k, threshold):
    if k > 0:
        filtered_data = data[data[attributes[0] + "_Prev0"] != 0]
    else:
        filtered_data = data

    # Create all relations that can have a Functional Dependency
    mappings = []
    for attr1 in attributes:
        for attr2 in attributes:
            # Check attr2 -> attr1
            if attr1 != attr2 and is_mapping(filtered_data[attr1], filtered_data[attr2], threshold):
                mappings.append((attr2, attr1))
            for i in range(k):
                prev_attr2 = attr2 + "_Prev%i" % (i)
                # Check attr2 -> attr1
                if is_mapping(filtered_data[attr1], filtered_data[prev_attr2], threshold):
                    mappings.append((prev_attr2, attr1))

    # Filter out mappings between previousA and currentB if currentA already maps to currentB and previousA to currentA
    #out_mappings = []
    #for mapping in mappings:
    #    found = 0
    #    from_index = mapping[0]
    #    to_index = mapping[1]
    #    if from_index.startswith("Prev"):
    #        check_from = from_index.replace("Prev0_", "") # TODO make more general with regex
    #        for check_mappings in mappings:
    #            if (check_mappings[0] == check_from and check_mappings[1] == to_index) or (check_mappings[0] == from_index and check_mappings[1] == check_from):
    #                found +=1
    #    if found < 2:
    #        out_mappings.append((from_index, to_index))
    #return out_mappings
    return mappings

    for column_from in data:
        for column_to in data:
            # Only consider different columns where TO is not a previous variable
            if column_to != column_from and (not previous_vals or data.columns.get_loc(column_to) - num_attrs >= 0) and is_mapping(data[column_to], data[column_from], 0.95):
                        mappings.append((data.columns.get_loc(column_from), data.columns.get_loc(column_to)))
    named_mappings = []
    if previous_vals:
        # Filter out mappings between previousA and currentB if currentA already maps to currentB and previousA to currentA
        for mapping in mappings:
            found = 0
            from_index = mapping[0]
            to_index = mapping[1]
            # Only check if from_index is previous
            if (from_index < num_attrs):
                check_from = from_index + num_attrs
                for check_mappings in mappings:
                    if (check_mappings[0] == check_from and check_mappings[1] == to_index) or (check_mappings[0] == from_index and check_mappings[1] == check_from):
                        found += 1
            if found < 2:
                named_mappings.append((data.columns[from_index], data.columns[to_index]))
    else:
        for mapping in mappings:
            named_mappings.append((data.columns[mapping[0]], data.columns[mapping[1]]))
    # Return list of 2-elem list, first element in 2-elem list is From_Index, second is To_Index
    # Return Indexes using original name from data
    return named_mappings

def calculate_new_values_rate(col):
    #return len(col[int(len(col)*0.75):].unique()) / len(col.unique())
    print(len(col.unique()), len(col))
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
    data = pd.read_csv("/Users/Stephen/Adrem_data/Data/example_input.csv", delimiter=",", nrows=10000, header=None, dtype=str, skiprows=0)
    data.columns = ["pTime","pEventID","pType","pActivity","pUserID","pUserName","pUserRole","pTrace","pAnoms","time","eventID","Type","Activity","UserID","UserName","UserRole","trace","anoms"]
    for from_attr in data:
        if from_attr not in ["pTime", "time", "pEventID", "eventID", "pTrace", "trace", "pAnoms", "anoms"]:
            row = []
            row.append(from_attr)
            for to_attr in data:
                if to_attr not in ["pTime", "time", "pEventID", "eventID", "pTrace", "trace", "pAnoms", "anoms"]:
                    #print(from_attr, to_attr, calculate_mutual_information(data[from_attr], data[to_attr]) / calculate_entropy(data[from_attr]))
                    row.append(str(round(calculate_mutual_information(data[to_attr], data[from_attr]) / calculate_entropy(data[to_attr]), 2)))
            print(" & ".join(row), "\\\\ \\hline")


