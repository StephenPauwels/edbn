import multiprocessing

import pandas as pd
from joblib import Parallel, delayed


# Constraint Bayesian Networks (CBN)
# Open-Domain: new values may be encountered
# Constraint: some of the mappings can be more strict (always map to the same values etc)
class ConstraintBayesianNetwork():
    def __init__(self, num_attrs, k, trace_attr, label_attr_nr, normal_label):
        self.variables = {}
        self.num_attrs = num_attrs
        self.label_attr_nr = label_attr_nr
        self.normal_label = normal_label
        self.log = None
        self.k = k
        self.trace_attr = trace_attr

    def add_variable(self, attr_id, name, new_values):
        self.variables[name] = Variable(attr_id, name, new_values, self.num_attrs)

    def remove_variable(self, name):
        del self.variables[name]

    def add_mapping(self, map_from_var, map_to_var):
        self.variables[map_to_var].add_mapping(self.variables[map_from_var])

    def add_parent(self, parent_var, attr_name):
        self.variables[attr_name].add_parent(self.variables[parent_var])

    def iterate_variables(self):
        for key in self.variables:
            yield (key, self.variables[key])

    def train(self, filename, delim, length):
        print("Training Network ...")

        data = pd.read_csv(filename, delimiter=delim, nrows=length, header=0, skiprows=0, dtype=str)
        self.log = self.create_k_context(data)

        for (_, value) in self.iterate_variables():
            self.train_var(value)
        print("Training Done")

    def create_k_context(self, data):
        traces = data.groupby([self.trace_attr])
        contextdata = {}
        # Get Attributes
        attributes = list(data.columns)
        for attribute in attributes:
            contextdata[attribute] = []
            for i in range(self.k):
                contextdata["Prev%i_" % (i) + attribute] = []
        for trace in traces:
            datatrace = trace[1]
            for event in range(len(datatrace)):
                for attr in range(len(attributes)):
                    contextdata[attributes[attr]].append(datatrace.iloc[event, attr])
                    for i in range(self.k):
                        if event - 1 - i < 0:
                            contextdata["Prev%i_" % (i) + attributes[attr]].append("0")
                        else:
                            contextdata["Prev%i_" % (i) + attributes[attr]].append(datatrace.iloc[event - 1 - i, attr])
        return pd.DataFrame(data=contextdata)

    def train_from_data(self, data):
        self.log = data
        for (_, value) in self.iterate_variables():
            self.train_var(value)
        print("Training Done")

    def train_var(self, var):
        print("Training", var.attr_name)
        var.train_variable(self.log)
        var.train_fdt(self.log)
        var.train_cpt(self.log)
        var.set_new_relation(self.log)
        return var

    def row_probability(self, row):
        prob = 1
        for (key, value) in self.iterate_variables():
            prob *= value.test_fdt(row) * value.test_cpt(row) * value.test_value(row)
        return (prob, row)

    def row_probability_detail(self, row):
        probs = []
        for (key, value) in self.iterate_variables():
            probs.append(value.test_fdt(row) * value.test_cpt(row) * value.test_value(row))
        return (probs, row)

    def row_scores_detail(self, row_list, labeled = False):
        probs = {}
        for (key, value) in self.iterate_variables():
            probs[key] = value.detailed_score(row_list)
        if labeled:
            probs["Total"] = {}
            probs["Total"]["anom"] = row_list[-1]
        return probs

    def get_variables(self):
        vars = []
        for (key, value) in self.iterate_variables():
            vars.append(value)

    def get_anomalies_sorted(self, filename, delim, length, skip):
        print("Sorting anomalies")
        data = pd.read_csv(filename, delimiter=delim, nrows=length, header=0, dtype=str, skiprows=skip)

        log = self.create_k_context(data)

        ranking = []
        for row in log.itertuples():
            ranking.append(self.row_probability_detail(row))
        ranking.sort(key=lambda l: l[0])
        return ranking

    def get_anomalies_sorted_parallel(self, filename, delim, length, skip):
        print("Sorting anomalies Parallel")
        njobs = multiprocessing.cpu_count()
        if length < njobs:
            part_length = length
            njobs = 1
        else:
            part_length = int(length / njobs)
        skips = [int((i * part_length)) + skip for i in range(0, njobs)]
        print(skips)
        results = []
        print(length, njobs, part_length)
        for r in Parallel(n_jobs=njobs)(delayed(self.get_anomalies_sorted)(filename, delim, part_length, s) for s in skips):
            results.extend(r)
        results.sort(key=lambda l: l[0])
        return results

    def get_scores_detail(self, filename, delim, length, skip):
        print("Sorting anomalies")
        log = pd.read_csv(filename, delimiter=delim, nrows=length, header=0, dtype=str, skiprows=skip)
        ranking = []
        for row in log.itertuples():
            ranking.append(self.row_scores_detail(list(row)[1:]))
        return ranking

    def get_scores_detail_parallel(self, filename, delim, length, skip):
        print("Calculating scores Parallel")
        njobs = multiprocessing.cpu_count()
        if length < njobs:
            part_length = length
            njobs = 1
        else:
            part_length = int(length / njobs)
        skips = [int((i * part_length)) + skip for i in range(0, njobs)]
        print(skips)
        results = []
        print(length, njobs, part_length)
        for r in Parallel(n_jobs=njobs)(delayed(self.get_scores_detail)(filename, delim, part_length, s) for s in skips):
            results.extend(r)
        return results


    def check_top_k(self, anomalies, k):
        false_errors = 0
        true_errors = 0
        k = min(k, len(anomalies))
        for i in range(k):
            if anomalies[i][1][self.label_attr_nr] == self.normal_label:
                false_errors += 1
            else:
                true_errors += 1
        print("Top-" + str(k) + " | True Errors found:", true_errors, "% | False Errors found:", false_errors, "%")
        print("Top-" + str(k) + " | True Errors found:", (true_errors/k)*100, "% | False Errors found:", (false_errors/k)*100, "%")


    def write_to_file(self, filename):
        with open(filename, "w") as fout:
            fout.write(str(self.num_attrs) + ";" + str(self.label_attr_nr) + ";" + str(self.normal_label) + "\n")
            for (key,value) in self.iterate_variables():
                fout.write(str(key) + ";" + str(value.attr_id) + ";" + str(value.new_values) + ";" + str([a.attr_name for a in value.conditional_parents]) + ";" + str([a.attr_name for a in value.functional_parents]) + "\n")

    def load_from_file(self, filename):
        conditional_parents = {}
        functional_parents = {}
        with open(filename, "r") as fin:
            # Read Model
            model_desc = fin.readline().split(";")
            self.num_attrs = int(model_desc[0])
            self.label_attr_nr = int(model_desc[1])
            self.normal_label = model_desc[2]
            for l in fin:
                var_desc = l.split(";")
                name = var_desc[0]
                self.add_variable(int(var_desc[1]), name, float(var_desc[2]))
                cond_parents = eval(var_desc[3])
                print("Cond:", cond_parents)
                conditional_parents[name] = []
                for p in cond_parents:
                    conditional_parents[name].append(p)

                func_parents = eval(var_desc[4])
                functional_parents[name] = []
                for p in func_parents:
                    functional_parents[name].append(p)

        for k in conditional_parents:
            for p in conditional_parents[k]:
                self.add_parent(p, k)
        for k in functional_parents:
            for p in functional_parents[k]:
                self.add_mapping(p, k)


class Variable:
    def __init__(self, attr_id, attr_name, new_values, num_attrs):
        self.attr_id = attr_id
        self.attr_name = attr_name
        self.new_values = new_values
        self.new_relations = 0
        self.num_attrs = num_attrs

        self.values = set()

        self.conditional_parents = []
        self.cpt = dict()
        self.functional_parents = []
        self.fdt = []

    def __repr__(self):
        return str(self.attr_id) + " - " + self.attr_name

    def add_parent(self, var):
        self.conditional_parents.append(var)

    def add_mapping(self, var):
        self.functional_parents.append(var)
        self.fdt.append({})

    def test_consistency(self):
        print("Consistent?", self.attr_id,":", len(self.cpt), len(self.fdt))

    def set_new_relation(self, log):
        attrs = set()
        attrs = attrs.union({p.attr_name for p in self.conditional_parents}).union({self.attr_name})
        grouped = log.groupby([a for a in attrs]).size().reset_index(name='counts')
        self.new_relations = len(grouped) / log.shape[0]
        print("NEW RELATION:", len(grouped), log.shape[0])

    def train_variable(self, log):
        self.values = set(log[self.attr_name].unique())

    def train_fdt(self, log):
        if len(self.functional_parents) == 0:
            return

        for i in range(len(self.functional_parents)):
            log_size = log.shape[0]
            parent = self.functional_parents[i]
            grouped = log.groupby([parent.attr_name, self.attr_name]).size().reset_index(name='counts')
            tmp_mapping = {}
            for t in grouped.itertuples():
                row = list(t)
                parent_val = row[1]
                val = row[2]
                if parent_val not in tmp_mapping or row[-1] > tmp_mapping[parent_val][0]:
                    tmp_mapping[parent_val] = (row[-1], val)

            for p in tmp_mapping:
                self.fdt[i][p] = tmp_mapping[p][1]

    def train_cpt(self, log):
        if len(self.conditional_parents) == 0:
            return

        log_size = log.shape[0]
#        print("Training variable", self.attr_id, "with parents", [p.attr_id for p in self.conditional_parents])
        grouped = log.groupby([p.attr_name for p in self.conditional_parents] + [self.attr_name]).size().reset_index(name='counts')
        parent_grouped = log.groupby([p.attr_name for p in self.conditional_parents]).size().reset_index(name='counts')
        for t in grouped.itertuples():
            row = list(t)
            parent_size = 0
            for p_g in parent_grouped.itertuples():
                p_row = list(p_g)
                if "-".join(row[1:-2]) == "-".join(p_row[1:-1]):
                    parent_size = p_row[-1]
            # Set to probability of getting value when parent configuration is seen: P( X | Parent(X))
            self.cpt["-".join(row[1:-1])] = row[-1] / parent_size

    def detailed_score(self, row):
        prob_result = {}
        prob_result["value"] = self.test_value(row)
        if len(self.conditional_parents) > 0:
            prob_result["cpt"] = self.test_cpt(row)
        if len(self.functional_parents) > 0:
            prob_result["fdt"] = self.test_fdt(row)
        return prob_result

    def test_fdt(self, row):
        if len(self.functional_parents) > 0:
            for i in range(len(self.functional_parents)):
                parent = self.functional_parents[i]
                if getattr(row, parent.attr_name) not in self.fdt[i]:
                    return 1
                if self.fdt[i][getattr(row, parent.attr_name)] == getattr(row, self.attr_name) or getattr(row, parent.attr_name) == "0":
                    return 1
                else:
                    return 0
        return 1


    def test_cpt(self, row):
        if len(self.conditional_parents) > 0:
            parent_vals = []
            for idx in [p.attr_name for p in self.conditional_parents] + [self.attr_name]:
                parent_vals.append(str(getattr(row,idx)))
            parent_config = "-".join(parent_vals)

            if parent_config not in self.cpt:
                return self.new_relations

            # If CPT -> return value for CPT
            else:
                return self.cpt[parent_config]
        return 1

    def test_value(self, row):
        if getattr(row, self.attr_name) not in self.values:
            return self.new_values
        else:
            return 1
