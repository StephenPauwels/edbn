import pandas as pd
import multiprocessing as mp
from dateutil.parser import parse
from sklearn import preprocessing
from collections import defaultdict
import numpy as np

class LogFile:

    def __init__(self, filename, delim, header, rows, time_attr, trace_attr, activity_attr = None, values = None, convert2integers = True):
        self.filename = filename
        self.time = time_attr
        self.trace = trace_attr
        self.activity = activity_attr
        if values is not None:
            self.values = values
        else:
            self.values = {}

        self.data = pd.read_csv(self.filename, header=header, nrows=rows, delimiter=delim, dtype="str")

        if convert2integers:
            self.convert2ints("../converted_ints.csv")

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
        # TODO: fix issue as first column is processed twice
        self.data = self.data.apply(lambda x: self.convert_column2ints(x))
        self.data.to_csv(file_out, index=False)

    def convert_column2ints(self, x):
        if x.name not in self.values:
            self.values[x.name], y = np.unique(x, return_inverse=True)
            return y
        else:
            self.values[x.name] = np.append(self.values[x.name], np.setdiff1d(np.unique(x), self.values[x.name]))

            xsorted = np.argsort(self.values[x.name])
            ypos = np.searchsorted(self.values[x.name][xsorted], x)
            indices = xsorted[ypos]

            return indices

    def convert_string2int(self, column, value):
        vals = self.values[column]
        found = np.where(vals==value)
        if len(found[0]) == 0:
            return None
        else:
            return found[0][0]

    def convert_int2string(self, column, int_val):
        return self.values[column][int_val]



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



"""
def convert(col):
    string_2_int = {}
    new_col = []
    for i in range(len(col)):
        value = col.iloc[i]
        if value not in string_2_int:
            string_2_int[value] = len(string_2_int) + 1
        new_col.append(string_2_int[value])
    return pd.Series(new_col)
"""

if __name__ == "__main__":
    #log = LogFile("../Data/bpic2018.csv", ",", 0, 100, "startTime", "case")
    log = LogFile("../Data/BPIC15_train_1.csv", ",", 0, 100, "time", "Case")
    print(log.data)

    log2 = LogFile("../Data/BPIC15_test_1.csv", ",", 0, 100, "time", "Case", values=log.values)
    print(log2.data)

    #log.create_k_context()
    #log.add_duration_to_k_context()
    #data = pd.read_csv("../Data/bpic2018.csv", delimiter=",", header=0, nrows=3000, dtype=str)
    #print(data.swifter.apply(convert))
    #print(data.apply(convert, axis=0))
