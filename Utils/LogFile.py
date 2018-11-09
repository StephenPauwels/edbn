import pandas as pd
import multiprocessing as mp
from dateutil.parser import parse

class LogFile:

    def __init__(self, filename, delim, header, rows, time_attr, trace_attr, string_2_int = None, int_2_string = None):
        self.filename = filename
        self.time = time_attr
        self.trace = trace_attr
        if string_2_int is not None:
            self.string_2_int = string_2_int
            self.int_2_string = int_2_string
        else:
            self.string_2_int = {}
            self.int_2_string = {}
        self.convert2ints("../converted_ints.csv", delim, rows, header=True)
        self.data = pd.read_csv("../converted_ints.csv", delimiter=delim, header=header, nrows=rows, dtype=int)
        self.contextdata = None
        self.k = 1

    def convert2ints(self, file_out, delimiter, rows, header=True):
        """
        Convert csv file with string values to csv file with integer values.
        (File/string operations more efficient than pandas operations)

        :param file_out: filename for newly created file
        :param delimiter: delimiter of input file, gets used by output file
        :param rows: number of rows to convert
        :param header: header present in the input?
        :return: number of lines converted
        """
        cnt = 0

        with open(self.filename, "r") as fin:
            with open(file_out, "w") as fout:
                header_list = None
                if header:
                    header_line = fin.readline()
                    fout.write(header_line)
                    header_list = header_line.replace('"', '').replace("\n", "").split(delimiter)
                    for attr in header_list:
                        if attr not in self.string_2_int:
                            self.string_2_int[attr] = {}
                            self.int_2_string[attr] = {}

                for line in fin:
                    cnt += 1
                    input = line.replace('"', '').replace("\n", "").split(delimiter)
                    output = []
                    attr = 0
                    for i in input:
                        attribute = header_list[attr]
                        if i not in self.string_2_int[attribute]:
                            int_val = len(self.string_2_int[attribute]) + 1
                            self.string_2_int[attribute][i] = int_val
                            self.int_2_string[attribute][int_val] = i
                        output.append(str(self.string_2_int[attribute][i]))
                        attr += 1
                    fout.write(delimiter.join(output))
                    fout.write("\n")
                    if cnt > rows:
                        break
        return cnt


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
            startTime = parse(self.int_2_string[self.time][row[self.time + "_Prev%i" % (k)]])
            endTime = parse(self.int_2_string[self.time][row[self.time]])
            return (endTime - startTime).total_seconds()
        else:
            return 0


def convert(col):
    string_2_int = {}
    new_col = []
    for i in range(len(col)):
        value = col.iloc[i]
        if value not in string_2_int:
            string_2_int[value] = len(string_2_int) + 1
        new_col.append(string_2_int[value])
    return pd.Series(new_col)


if __name__ == "__main__":
    log = LogFile("../Data/bpic2018.csv", ",", 0, 3000, "startTime", "case")
    #log.create_k_context()
    #log.add_duration_to_k_context()
    #data = pd.read_csv("../Data/bpic2018.csv", delimiter=",", header=0, nrows=3000, dtype=str)
    #print(data.swifter.apply(convert))
    #print(data.apply(convert, axis=0))