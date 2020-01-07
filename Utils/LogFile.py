import copy
import multiprocessing as mp

import numpy as np
import pandas as pd
from dateutil.parser import parse


class LogFile:

    def __init__(self, filename, delim, header, rows, time_attr, trace_attr, activity_attr = None, values = None, integer_input = False, convert = True, k = 1):
        self.filename = filename
        self.time = time_attr
        self.trace = trace_attr
        self.activity = activity_attr
        if values is not None:
            self.values = values
        else:
            self.values = {}
        self.numericalAttributes = set()
        self.categoricalAttributes = set()
        if self.trace is None:
            self.k = 0
        else:
            self.k = k

        type = "str"
        if integer_input:
            type = "int"
        if filename is not None:
            self.data = pd.read_csv(self.filename, header=header, nrows=rows, delimiter=delim, encoding='latin-1', dtype="str")

            # Determine types for all columns - numerical or categorical
            for col_type in self.data.dtypes.iteritems():
                #if col_type[1] == 'float64':
                #    self.numericalAttributes.add(col_type[0])
                #else:
                    self.categoricalAttributes.add(col_type[0])

            if convert:
                self.convert2int()

            self.contextdata = None

    def get_data(self):
        if self.contextdata is None:
            return self.data
        return self.contextdata

    def get_cases(self):
        return self.get_data().groupby([self.trace])
    
    def filter_case_length(self, min_length):
        cases = self.data.groupby([self.trace])
        filtered_cases = []
        for case in cases:
            if len(case[1]) > min_length:
                filtered_cases.append(case[1])
        self.data = pd.concat(filtered_cases, ignore_index=True)

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

        if self.isNumericAttribute(x.name):
            return x


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
        if column not in self.values:
            return value
        vals = self.values[column]
        found = np.where(vals==value)
        if len(found[0]) == 0:
            return None
        else:
            return found[0][0] + 1

    def convert_int2string(self, column, int_val):
        if column not in self.values:
            return int_val
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

    def filter_copy(self, filter_condition):
        log_copy = copy.deepcopy(self)
        log_copy.data = self.data[eval(filter_condition)]
        return log_copy

    def get_column(self, attribute):
        return self.data[attribute]

    def get_labels(self, label):
        labels = {}
        if self.trace is None:
            for row in self.data.itertuples():
                labels[row.Index] = getattr(row, label)
        else:
            traces = self.data.groupby([self.trace])
            for trace in traces:
                labels[trace[0]] = getattr(trace[1].iloc[0], label)
        return labels

    def create_k_context(self):
        """
        Create the k-context from the current LogFile

        :return: None
        """
        print("Create k-context:", self.k)

        if self.k == 0:
            self.contextdata = self.data

        if self.contextdata is None:
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
            self.numericalAttributes.add("duration_%i" % (i))

    def calc_duration(self, row, k):
        if row[self.time + "_Prev%i" % (k)] != 0:
            startTime = parse(self.convert_int2string(self.time, int(row[self.time + "_Prev%i" % (k)])))
            endTime = parse(self.convert_int2string(self.time,int(row[self.time])))
            return (endTime - startTime).total_seconds()
        else:
            return 0

    def discretize(self,row, bins=25):
        if isinstance(bins, int):
            labels = [str(i) for i in range(1,bins+1)]
        else:
            labels = [str(i) for i in range(1,len(bins))]
        if self.isNumericAttribute(row):
            self.numericalAttributes.remove(row)
            self.categoricalAttributes.add(row)
            self.contextdata[row], binned = pd.cut(self.contextdata[row], bins, retbins=True, labels=labels)
            #self.contextdata[row] = self.contextdata[row].astype(str)
            #self.contextdata[row] = self.convert_column2ints(self.contextdata[row])
        return binned

    def isNumericAttribute(self, attribute):
        if attribute in self.numericalAttributes:
            return True
        else:
            for k in range(self.k):
                if attribute.replace("_Prev%i" % (k), "") in self.numericalAttributes:
                    return True
        return False

    def isCategoricalAttribute(self, attribute):
        if attribute in self.categoricalAttributes:
            return True
        else:
            for k in range(self.k):
                if attribute.replace("_Prev%i" % (k), "") in self.categoricalAttributes:
                    return True
        return False

    def splitTrainTest(self, train_percentage):
        import random

        grouped = self.get_data().groupby([self.trace])
        train_cases = []
        test_cases = []
        for group in grouped:
            if random.randint(0,100) < train_percentage:
                train_cases.append(group[1])
            else:
                test_cases.append(group[1])

        train = pd.concat(train_cases)
        test = pd.concat(test_cases)

        train_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
        train_logfile.filename = self.filename
        train_logfile.values = self.values
        train_logfile.contextdata = train
        train_logfile.categoricalAttributes = self.categoricalAttributes
        train_logfile.numericalAttributes = self.numericalAttributes
        train_logfile.data = self.data
        train_logfile.k = self.k

        test_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
        test_logfile.filename = self.filename
        test_logfile.values = self.values
        test_logfile.contextdata = test
        test_logfile.categoricalAttributes = self.categoricalAttributes
        test_logfile.numericalAttributes = self.numericalAttributes
        test_logfile.data = self.data
        test_logfile.k = self.k

        return train_logfile, test_logfile