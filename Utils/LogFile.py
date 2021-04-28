"""
    Author: Stephen Pauwels
"""

import copy
import multiprocessing as mp

import numpy as np
import pandas as pd
from dateutil.parser import parse
import math


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
        self.ignoreHistoryAttributes = set()
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

        if self.time is not None and x.name == self.time:
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
        if self.time and self.time not in keep_attrs and self.time in self.data:
            keep_attrs.append(self.time)
        if self.trace and self.trace not in keep_attrs:
            keep_attrs.append(self.trace)
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

    def create_trace_attribute(self):
        print("Create trace attribute")
        with mp.Pool(mp.cpu_count()) as p:
            result = p.map(self.create_trace_attribute_case, self.data.groupby([self.trace]))
        self.data = pd.concat(result)
        self.categoricalAttributes.add("trace")

    def create_trace_attribute_case(self, case_tuple):
        trace = []
        case_data = pd.DataFrame()
        for row in case_tuple[1].iterrows():
            row_content = row[1]
            trace.append(row_content[self.activity])
            row_content["trace"] = str(trace)
            case_data = case_data.append(row_content)
        return case_data

    def create_k_context(self):
        """
        Create the k-context from the current LogFile

        :return: None
        """
        print("Create k-context:", self.k)

        if self.k == 0:
            self.contextdata = self.data

        if self.contextdata is None:
            # result = map(self.create_k_context_trace, self.data.groupby([self.trace]))

            with mp.Pool(mp.cpu_count()) as p:
                result = p.map(self.create_k_context_trace, self.data.groupby([self.trace]))

            # result = map(self.create_k_context_trace, self.data.groupby([self.trace]))

            self.contextdata = pd.concat(result, ignore_index=True)

    def create_k_context_trace(self, trace):
        contextdata = pd.DataFrame()

        trace_data = trace[1]
        shift_data = trace_data.shift().fillna(0)
        shift_data.at[shift_data.first_valid_index(), self.trace] = trace[0]
        joined_trace = shift_data.join(trace_data, lsuffix="_Prev0")
        for i in range(1, self.k):
            shift_data = shift_data.shift().fillna(0)
            shift_data.at[shift_data.first_valid_index(), self.trace] = trace[0]
            joined_trace = shift_data.join(joined_trace, lsuffix="_Prev%i" % i)
        contextdata = contextdata.append(joined_trace, ignore_index=True)
        contextdata = contextdata.astype("int", errors="ignore")
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

    def add_end_events(self):
        cases = self.get_cases()
        print("Run end event map")
        with mp.Pool(mp.cpu_count()) as p:
            result = p.map(self.add_end_event_case, cases)

        print("Combine results")
        new_data = []
        for r in result:
            new_data.extend(r)

        self.data = pd.DataFrame.from_records(new_data)

    def add_end_event_case(self, case_obj):
        case_name, case = case_obj
        new_data = []
        for i in range(0, len(case)):
            new_data.append(case.iloc[i].to_dict())

        record = {}
        for col in self.data:
            if col == self.trace:
                record[col] = case_name
            elif col == self.time:
                record[col] = new_data[-1][self.time]
            else:
                record[col] = "end"
        new_data.append(record)
        return new_data

    def splitTrainTest(self, train_percentage, split_case, method="random"):
        import random
        train_percentage = train_percentage / 100.0

        if split_case:
            if method == "random":
                train_inds = random.sample(range(self.contextdata.shape[0]), k=round(self.contextdata.shape[0] * train_percentage))
                test_inds = list(set(range(self.contextdata.shape[0])).difference(set(train_inds)))
            elif method == "train-test":
                train_inds = np.arange(0, self.contextdata.shape[0] * train_percentage)
                test_inds = list(set(range(self.contextdata.shape[0])).difference(set(train_inds)))
            else:
                test_inds = np.arange(0, self.contextdata.shape[0] * (1 - train_percentage))
                train_inds = list(set(range(self.contextdata.shape[0])).difference(set(test_inds)))
        else:
            train_inds = []
            test_inds = []
            cases = self.contextdata[self.trace].unique()
            if method == "random":
                train_cases = random.sample(list(cases), k=round(len(cases) * train_percentage))
                test_cases = list(set(cases).difference(set(train_cases)))
            elif method == "train-test":
                train_cases = cases[:round(len(cases) * train_percentage)]
                test_cases = cases[round(len(cases) * train_percentage):]
            else:
                train_cases = cases[round(len(cases) * (1 - train_percentage)):]
                test_cases = cases[:round(len(cases) * (1 - train_percentage))]

            for train_case in train_cases:
                train_inds.extend(list(self.contextdata[self.contextdata[self.trace] == train_case].index))
            for test_case in test_cases:
                test_inds.extend(list(self.contextdata[self.contextdata[self.trace] == test_case].index))

        train = self.contextdata.loc[train_inds]
        test = self.contextdata.loc[test_inds]

        print("Train:", len(train_inds))
        print("Test:", len(test_inds))

        train_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
        train_logfile.filename = self.filename
        train_logfile.values = self.values
        train_logfile.contextdata = train
        train_logfile.categoricalAttributes = self.categoricalAttributes
        train_logfile.numericalAttributes = self.numericalAttributes
        train_logfile.data = self.data.loc[train_inds]
        train_logfile.k = self.k

        test_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
        test_logfile.filename = self.filename
        test_logfile.values = self.values
        test_logfile.contextdata = test
        test_logfile.categoricalAttributes = self.categoricalAttributes
        test_logfile.numericalAttributes = self.numericalAttributes
        test_logfile.data = self.data.loc[test_inds]
        test_logfile.k = self.k

        return train_logfile, test_logfile


    def split_days(self, date_format, num_days=1):
        from datetime import datetime

        self.contextdata["days"] = self.contextdata[self.time].map(lambda l: str(datetime.strptime(l, date_format).isocalendar()[:3]))
        days = {}
        for group_name, group in self.contextdata.groupby("days"):
            new_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
            new_logfile.filename = self.filename
            new_logfile.values = self.values
            new_logfile.categoricalAttributes = self.categoricalAttributes
            new_logfile.numericalAttributes = self.numericalAttributes
            new_logfile.k = self.k
            new_logfile.contextdata = group.drop("days", axis=1)
            new_logfile.data = new_logfile.contextdata[self.attributes()]

            days[group_name] = {}
            days[group_name]["data"] = new_logfile
        return days

    def split_weeks(self, date_format, num_days=1):
        from datetime import datetime

        self.contextdata["year_week"] = self.contextdata[self.time].map(lambda l: str(datetime.strptime(l, date_format).isocalendar()[:2]))
        weeks = {}
        for group_name, group in self.contextdata.groupby("year_week"):
            new_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
            new_logfile.filename = self.filename
            new_logfile.values = self.values
            new_logfile.categoricalAttributes = self.categoricalAttributes
            new_logfile.numericalAttributes = self.numericalAttributes
            new_logfile.k = self.k
            new_logfile.contextdata = group.drop("year_week", axis=1)
            new_logfile.data = new_logfile.contextdata[self.attributes()]

            year, week = eval(group_name)
            group_name = "%i/" % year
            if week < 10:
                group_name += "0"
            group_name += str(week)

            weeks[group_name] = {}
            weeks[group_name]["data"] = new_logfile
        return weeks

    def split_months(self, date_format, num_days=1):
        from datetime import datetime
        self.contextdata["month"] = self.contextdata[self.time].map(lambda l: str(datetime.strptime(l, date_format).strftime("%Y/%m")))

        months = {}
        for group_name, group in self.contextdata.groupby("month"):
            new_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
            new_logfile.filename = self.filename
            new_logfile.values = self.values
            new_logfile.categoricalAttributes = self.categoricalAttributes
            new_logfile.numericalAttributes = self.numericalAttributes
            new_logfile.k = self.k
            new_logfile.contextdata = group.drop("month", axis=1)
            new_logfile.data = new_logfile.contextdata[self.attributes()]

            months[group_name] = {}
            months[group_name]["data"] = new_logfile
        return months

    def split_date(self, date_format, year_week, from_week=None):
        from datetime import datetime

        self.contextdata["year_week"] = self.contextdata[self.time].map(lambda l: str(datetime.strptime(l, date_format).isocalendar()[:2]))

        if from_week:
            train = self.contextdata[(self.contextdata["year_week"] >= from_week) & (self.contextdata["year_week"] < year_week)]
        else:
            train = self.contextdata[self.contextdata["year_week"] < year_week]
        test = self.contextdata[self.contextdata["year_week"] == year_week]

        train_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
        train_logfile.filename = self.filename
        train_logfile.values = self.values
        train_logfile.contextdata = train
        train_logfile.categoricalAttributes = self.categoricalAttributes
        train_logfile.numericalAttributes = self.numericalAttributes
        train_logfile.data = train[self.attributes()]
        train_logfile.k = self.k

        test_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
        test_logfile.filename = self.filename
        test_logfile.values = self.values
        test_logfile.contextdata = test
        test_logfile.categoricalAttributes = self.categoricalAttributes
        test_logfile.numericalAttributes = self.numericalAttributes
        test_logfile.data = test[self.attributes()]
        test_logfile.k = self.k

        return train_logfile, test_logfile


    def create_folds(self, k):
        result = []
        folds = np.array_split(np.arange(0, self.contextdata.shape[0]), k)
        for f in folds:
            fold_context = self.contextdata.loc[f]

            logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
            logfile.filename = self.filename
            logfile.values = self.values
            logfile.contextdata = fold_context
            logfile.categoricalAttributes = self.categoricalAttributes
            logfile.numericalAttributes = self.numericalAttributes
            logfile.data = self.data.loc[f]
            logfile.k = self.k
            result.append(logfile)
        return result

    def extend_data(self, log):
        train_logfile = LogFile(None, None, None, None, self.time, self.trace, self.activity, self.values, False, False)
        train_logfile.filename = self.filename
        train_logfile.values = self.values
        train_logfile.contextdata = self.contextdata.append(log.contextdata)
        train_logfile.categoricalAttributes = self.categoricalAttributes
        train_logfile.numericalAttributes = self.numericalAttributes
        train_logfile.data = self.data.append(log.data)
        train_logfile.k = self.k
        return train_logfile

    def get_traces(self):
        return [list(case[1][self.activity]) for case in self.get_cases()]

    def get_follows_relations(self, window=None):
        return self.get_traces_follows_relations(self.get_traces(), window)

    def get_traces_follows_relations(self, traces, window):
        follow_counts = {}
        counts = {}
        for trace in traces:
            for i in range(len(trace)):
                act = trace[i]
                if act not in follow_counts:
                    follow_counts[act] = {}
                    counts[act] = 0
                counts[act] += 1

                stop_value = len(trace)
                if window:
                    stop_value = min(len(trace), i+window)

                for fol_act in set(trace[i+1:stop_value+1]):
                    if fol_act not in follow_counts[act]:
                        follow_counts[act][fol_act] = 0
                    follow_counts[act][fol_act] += 1


        follows = {}
        for a in range(1, len(self.values[self.activity])+1):
            always = 0
            sometimes = 0
            if a in follow_counts:
                for b in follow_counts[a]:
                    if a != b:
                        if follow_counts[a][b] == counts[a]:
                            always += 1
                        else:
                            sometimes += 1
            never = len(self.values[self.activity]) - always - sometimes
            follows[a] = (always, sometimes, never)

        return follows, follow_counts


    def get_relation_entropy(self):
        follows, _ = self.get_follows_relations()
        full_entropy = []
        for act in range(1, len(self.values[self.activity])+1):
            RC = follows[act]
            p_a = RC[0] / len(self.values[self.activity])
            p_s = RC[1] / len(self.values[self.activity])
            p_n = RC[2] / len(self.values[self.activity])
            entropy = 0
            if p_a != 0:
                entropy -= p_a * math.log(p_a)
            if p_s != 0:
                entropy -= p_s * math.log(p_s)
            if p_n != 0:
                entropy -= p_n * math.log(p_n)
            full_entropy.append(entropy)
        return full_entropy


    def get_j_measure_trace(self, trace, window):
        _, follows = self.get_traces_follows_relations([trace], window)
        j_measure = []
        value_counts = {}
        for e in trace:
            if e not in value_counts:
                value_counts[e] = 0
            value_counts[e] += 1
        for act_1 in range(1, len(self.values[self.activity])+1):
            for act_2 in range(1, len(self.values[self.activity]) + 1):
                num_events = len(trace)
                if act_1 in follows and act_2 in follows[act_1]:
                    p_aFb = follows[act_1][act_2] / value_counts.get(act_1, 0)
                else:
                    p_aFb = 0

                if act_1 not in value_counts:
                    p_a = 0
                else:
                    p_a = value_counts.get(act_1, 0)/ num_events

                if act_2 not in value_counts:
                    p_b = 0
                else:
                    p_b = value_counts.get(act_2, 0) / num_events

                j_value = 0
                if p_aFb != 0 and p_b != 0:
                    j_value += p_aFb * math.log(p_aFb / p_b, 2)

                if p_aFb != 1 and p_b != 1:
                    j_value += (1-p_aFb) * math.log((1-p_aFb) / (1-p_b), 2)

                j_measure.append(p_a * j_value)

        return j_measure


    def get_j_measure(self, window=5):
        traces = self.get_traces()
        # return [np.mean(self.get_j_measure_trace(trace, window)) for trace in traces]
        return [self.get_j_measure_trace(trace, window) for trace in traces]
        # j_measures = np.asarray([self.get_j_measure_trace(trace, window) for trace in traces])
        # avg_j_measures = [np.mean(j_measures[:,i]) for i in range(len(j_measures[0]))]
        # return avg_j_measures

