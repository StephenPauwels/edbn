import pandas as pd
import multiprocessing as mp
from dateutil.parser import parse
from sklearn import preprocessing
from collections import defaultdict
import numpy as np

class LogFile:

    def __init__(self, filename, delim, header, rows, time_attr, trace_attr, activity_attr = None, values = None, integer_input = False, convert = True):
        self.filename = filename
        self.time = time_attr
        self.trace = trace_attr
        self.activity = activity_attr
        if values is not None:
            self.values = values
        else:
            self.values = {}

        type = "str"
        if integer_input:
            type = "int"
        self.data = pd.read_csv(self.filename, header=header, nrows=rows, delimiter=delim, dtype=type)

        if convert:
            self.convert2int()

        self.contextdata = None
        self.k = 1

    def convert2int(self):
        self.convert2ints("../converted_ints.csv")

    def convert2ints(self, file_out):
        """
        Convert csv file with string values to csv file with integer values.
        (File/string operations more efficient than pandas operations)

        :param file_out: filename for newly created file
        :return: number of lines converted
        """
        self.data = self.data.apply(lambda x: self.convert_column2ints(x))
        self.data.to_csv(file_out, index=False)

    def convert_column2ints(self, x):

        def test(a, b):
            # Return all elements from a that are not in b, make use of the fact that both a and b are unique and sorted
            a_ix = 0
            b_ix = 0
            new_uniques = []
            while a_ix < len(a) and b_ix < len(b):
                if a[a_ix] < b[b_ix]:
                    new_uniques.append(a[a_ix])
                    a_ix += 1
                elif a[a_ix] > b[b_ix]:
                    b_ix += 1
                else:
                    a_ix += 1
                    b_ix += 1
            if a_ix < len(a):
                new_uniques.extend(a[a_ix:])
            return new_uniques


        print("PREPROCESSING: Converting", x.name)
        if x.name not in self.values:
            x = x.astype("str")
            self.values[x.name], y = np.unique(x, return_inverse=True)
            return y + 1
        else:
            x = x.astype("str")
            self.values[x.name] = np.append(self.values[x.name], test(np.unique(x), self.values[x.name]))

            print("PREPROCESSING: Substituting values with ints")
            xsorted = np.argsort(self.values[x.name])
            ypos = np.searchsorted(self.values[x.name][xsorted], x)
            indices = xsorted[ypos]

        return indices + 1

    def convert_string2int(self, column, value):
        vals = self.values[column]
        found = np.where(vals==value)
        if len(found[0]) == 0:
            return None
        else:
            return found[0][0] + 1

    def convert_int2string(self, column, int_val):
        return self.values[column][int_val - 1]


    def attributes(self):
        return self.data.columns

    def keep_attributes(self, keep_attrs):
        self.data = self.data[keep_attrs]

    def remove_attributes(self, remove_attrs):
        """
        Remove attributes with the given prefixes from the data

        :param remove_attrs: a list of prefixes of attributes that should be removed from the data
        :return: None
        """
        remove = []
        for attr in self.data:
            for prefix in remove_attrs:
                if attr.startswith(prefix):
                    remove.append(attr)
                    break

        self.data = self.data.drop(remove, axis=1)

    def filter(self, filter_condition):
        self.data = self.data[eval(filter_condition)]

    def get_column(self, attribute):
        return self.data[attribute]

    def create_k_context(self):
        """
        Create the k-context from the current LogFile

        :return: None
        """
        if self.contextdata is None:
            print("Start creating k-context Parallel")

            with mp.Pool(mp.cpu_count()) as p:
                result = p.map(self.create_k_context_trace, self.data.groupby([self.trace]))
            self.contextdata = pd.concat(result, ignore_index=True)

    def create_k_context_trace(self, trace):
        contextdata = pd.DataFrame()

        trace_data = trace[1]
        shift_data = trace_data.shift().fillna(0).astype(int)
        shift_data.at[shift_data.first_valid_index(), self.trace] = trace[0]
        joined_trace = shift_data.join(trace_data, lsuffix="_Prev0")
        for i in range(1, self.k):
            shift_data = shift_data.shift().fillna(0).astype(int)
            shift_data.at[shift_data.first_valid_index(), self.trace] = trace[0]
            joined_trace = shift_data.join(joined_trace, lsuffix="_Prev%i" % i)
        contextdata = contextdata.append(joined_trace, ignore_index=True)
        return contextdata

    def add_duration_to_k_context(self):
        """
        Add durations to the k-context, only calculates if k-context has been calculated

        :return:
        """
        if self.contextdata is None:
            return

        for i in range(self.k):
            self.contextdata['duration_%i' %(i)] = self.contextdata.apply(self.calc_duration, axis=1, args=(i,))

    def calc_duration(self, row, k):
        if row[self.time + "_Prev%i" % (k)] != 0:
            startTime = parse(self.convert_int2string(self.time, row[self.time + "_Prev%i" % (k)]))
            endTime = parse(self.convert_int2string(self.time,row[self.time]))
            return (endTime - startTime).total_seconds()
        else:
            return 0

if __name__ == "__main__":
    import Uncertainty_Coefficient as uc

    print("Read train")
    train = LogFile("../Data/bpic2018.csv", ",", 0, 30000, "startTime", "case", convert=False)
    train.remove_attributes(["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    train.convert2int()

    print("Read converted")
    converted_data = LogFile("../Data/bpic2018.csv", ",", 0, 5000000, "startTime", "case", convert=False, values=train.values)
    converted_data.remove_attributes(["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    converted_data.convert2int()

    print("Read int")
    int_data = LogFile("../Data/bpic2018_ints.csv", ",", 0, 5000000, "startTime", "case", convert=False, integer_input=True)
    int_data.remove_attributes(["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])

    for attr in converted_data.data:
        print("ATTRIBUTE", attr)
        print(uc.calculate_mutual_information(converted_data.data[attr], int_data.data[attr]))
        print(uc.calculate_entropy(converted_data.data[attr]),uc.calculate_entropy(int_data.data[attr]))
        print(uc.is_mapping(converted_data.data["case"], int_data.data["case"], 0.99))
