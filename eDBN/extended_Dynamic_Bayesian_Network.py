import multiprocessing as mp
import re
import math

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def process(trace):
    k_contexts = model.create_k_context_trace(trace)
    result = {}
    for row in k_contexts.itertuples():
        prob = model.row_probability(row)
        case = int(getattr(row, "case"))
        if case not in result:
            result[case] = []
        result[case].append(prob)
    return result

def process_detail(trace):
    def product(l):
        score = 1
        for e in l:
            score *= e
        return score

    k_contexts = model.create_k_context_trace(trace)
    result = {}
    for row in k_contexts.itertuples():
        prob = model.row_scores_detail(row)
        for a in prob:
            score = prob[a].get("cpt", 1) * prob[a].get("value", 1)
            for f in prob[a].get("fdt", {}):
                score *= prob[a]["fdt"][f]
            if a not in result:
                result[a] = []
            result[a].append(score)

    return_result = []
    for a in sorted(result.keys()):
        #s = sum(result[a])
        #if s != 0:
        #    return_result.append(math.log10(s / len(result[a])))
        prod = product(result[a])
        if prod != 0:
            return_result.append(math.log10(math.pow(prod, 1 / len(result[a]))))
        else:
            return_result.append(-5)
    return np.asarray(return_result)

# Constraint Bayesian Networks (CBN)
# Open-Domain: new values may be encountered
# Constraint: some of the mappings can be more strict (always map to the same values etc)
class extendedDynamicBayesianNetwork():
    def __init__(self, num_attrs, k, trace_attr, label_attr_nr, normal_label):
        self.variables = {}
        self.current_variables = []
        self.num_attrs = num_attrs
        self.label_attr_nr = label_attr_nr
        self.normal_label = normal_label
        self.log = None
        self.k = k
        self.trace_attr = trace_attr

    def add_variable(self, name, new_values):
        self.variables[name] = Variable(name, new_values, self.num_attrs)
        m = re.search(r'Prev\d+$', name)
        if m is None:
            self.current_variables.append(name)

    def remove_variable(self, name):
        del self.variables[name]

    def add_mapping(self, map_from_var, map_to_var):
        self.variables[map_to_var].add_mapping(self.variables[map_from_var])

    def add_parent(self, parent_var, attr_name):
        self.variables[attr_name].add_parent(self.variables[parent_var])

    def iterate_variables(self):
        for key in self.variables:
            yield (key, self.variables[key])

    def iterate_current_variables(self):
        for key in self.current_variables:
            yield (key, self.variables[key])

    def train(self, filename, delim, length):
        print("Training Network ...")

        data = pd.read_csv(filename, delimiter=delim, nrows=length, header=0, skiprows=0, dtype=int)
        self.log = self.create_k_context(data)

        for (_, value) in self.iterate_current_variables():
            self.train_var(value)
        print("Training Done")

    def train_data(self, data):
        self.log = self.create_k_context(data)

        for (_, value) in self.iterate_current_variables():
            self.train_var(value)
        print("Training Done")

    def calculate_scores(self, data, accum_attr = None):
        def initializer(init_model):
            global model
            model = init_model

        if accum_attr is None:
            accum_attr = self.trace_attr
        with mp.Pool(mp.cpu_count(), initializer, (self,)) as p:
            scores = p.map(process, data.groupby([accum_attr]))
        return_scores = {}
        for x in scores:
            return_scores.update(x)
        return return_scores

    def calculate_scores_detail(self, data, accum_attr = None):
        def initializer(init_model):
            global model
            model = init_model

        print("EVALUATION: calculate score details")
        if accum_attr is None:
            accum_attr = self.trace_attr
        trace_data = data.groupby([accum_attr])
        with mp.Pool(mp.cpu_count(), initializer, (self,)) as p:
            result = p.map(process_detail, trace_data)
        print("EVALUATION: Done")
        scores = {}
        attributes = sorted([attr for attr in data.columns if attr != accum_attr])
        for trace_scores in result:
            for a_ix in range(len(attributes)):
                if attributes[a_ix] not in scores:
                    scores[attributes[a_ix]] = []
                scores[attributes[a_ix]].append(trace_scores[a_ix])
        return scores

    def create_k_context(self, data):
        print("Start creating k-context Parallel")

        with mp.Pool(mp.cpu_count()) as p:
            result = p.map(self.create_k_context_trace, data.groupby([self.trace_attr]))
        contextdata = pd.concat(result)
        return contextdata

    def create_k_context_trace(self, trace):
        contextdata = pd.DataFrame()

        trace_data = trace[1]
        shift_data = trace_data.shift().fillna(0).astype(int)
        shift_data.at[shift_data.first_valid_index(), self.trace_attr] = trace[0]
        joined_trace = shift_data.join(trace_data, lsuffix="_Prev0")
        for i in range(1, self.k):
            shift_data = shift_data.shift().fillna(0).astype(int)
            shift_data.at[shift_data.first_valid_index(), self.trace_attr] = trace[0]
            joined_trace = shift_data.join(joined_trace, lsuffix="_Prev%i" % i)
        contextdata = contextdata.append(joined_trace, ignore_index=True)
        # TODO: add duration to every timestep

        return contextdata

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
        not_zero = True
        for (key, value) in self.iterate_current_variables():
            score_fdt = 1
            fdt_scores = value.test_fdt(row)
            for fdt_score in fdt_scores:
                score_fdt *= fdt_scores[fdt_score]
            prob *= score_fdt * value.test_cpt(row) * value.test_value(row)
        return prob

    def row_probability_detail(self, row):
        probs = []
        for (key, value) in self.iterate_current_variables():
            score_fdt = 1
            fdt_scores = value.test_fdt(row)
            for fdt_score in fdt_scores:
                score_fdt *= fdt_scores[fdt_score]
            probs.append(score_fdt * value.test_cpt(row) * value.test_value(row))
        return probs

    def row_scores_detail(self, row_list, labeled = False):
        probs = {}
        for (key, value) in self.iterate_current_variables():
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
        data = pd.read_csv(filename, delimiter=delim, nrows=length, header=0, dtype=int, skiprows=skip)

        log = self.create_k_context(data)

        ranking = []
        for row in log.itertuples():
            ranking.append((self.row_probability_detail(row), row))
        ranking.sort(key=lambda l: l[0])
        return ranking

    def get_anomalies_sorted_parallel(self, filename, delim, length, skip):
        print("Sorting anomalies Parallel")
        njobs = mp.cpu_count()
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
        njobs = mp.cpu_count()
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
    def __init__(self, attr_name, new_values, num_attrs):
        self.attr_name = attr_name
        self.new_values = new_values
        self.new_relations = 0
        self.num_attrs = num_attrs

        self.values = set()

        self.conditional_parents = []
        self.cpt = dict()
        self.functional_parents = []
        self.fdt = []
        self.fdt_violation = []

    def __repr__(self):
        return self.attr_name

    def add_parent(self, var):
        self.conditional_parents.append(var)

    def add_mapping(self, var):
        self.functional_parents.append(var)
        self.fdt.append({})

    def test_consistency(self):
        print("Consistent?", self.attr_name,":", len(self.cpt), len(self.fdt))

    def set_new_relation(self, log):
        attrs = set()
        if len(self.conditional_parents) == 0:
            self.new_relations = 1
            return
        attrs = {p.attr_name for p in self.conditional_parents}
        grouped = log.groupby([a for a in attrs]).size().reset_index(name='counts')
        self.new_relations = len(grouped) / log.shape[0]

    def train_variable(self, log):
        self.values = set(log[self.attr_name].unique())

    def train_fdt(self, log):
        if len(self.functional_parents) == 0:
            return

        for i in range(len(self.functional_parents)):
            violations = 0
            log_size = log.shape[0]
            parent = self.functional_parents[i]
            grouped = log.groupby([parent.attr_name, self.attr_name]).size().reset_index(name='counts')
            tmp_mapping = {}
            for t in grouped.itertuples():
                row = list(t)
                parent_val = row[1]
                val = row[2]
                if parent_val not in tmp_mapping:
                    tmp_mapping[parent_val] = (row[-1], val)
                elif row[-1] > tmp_mapping[parent_val][0]:
                    violations += tmp_mapping[parent_val][0] # Add previous number to violations
                    tmp_mapping[parent_val] = (row[-1], val)
                else:
                    violations += row[-1] # Add number to violations

            for p in tmp_mapping:
                self.fdt[i][p] = tmp_mapping[p][1]

            self.fdt_violation.append(violations / log_size)
            print(self.functional_parents[i].attr_name, violations / log_size)

    def train_cpt(self, log):
        if len(self.conditional_parents) == 0:
            return

        parents = [p.attr_name for p in self.conditional_parents]
        grouped = log.groupby(parents)[self.attr_name]
        val_counts = grouped.value_counts()
        div = grouped.count().to_dict()
        for t in val_counts.items():
            parent = t[0][:-1]
            if len(parent) == 1:
                parent = parent[0]
            if parent not in self.cpt:
                self.cpt[parent] = dict()
            self.cpt[parent][t[0][-1]] = t[1] / div[parent]

    def detailed_score(self, row):
        prob_result = {}
        prob_result["value"] = self.test_value(row)
        if len(self.conditional_parents) > 0:
            prob_result["cpt"] = self.test_cpt(row)
        if len(self.functional_parents) > 0:
            prob_result["fdt"] = self.test_fdt(row)
        return prob_result

    def test_fdt(self, row):
        scores = {}
        if len(self.functional_parents) > 0:
            for i in range(len(self.functional_parents)):
                parent = self.functional_parents[i]
                if getattr(row, parent.attr_name) not in self.fdt[i]:
                    scores[parent.attr_name] = 1 - self.fdt_violation[i]
                    self.fdt[i][getattr(row, parent.attr_name)] = getattr(row, self.attr_name)
                elif self.fdt[i][getattr(row, parent.attr_name)] == getattr(row, self.attr_name) or getattr(row, parent.attr_name) == 0:
                    scores[parent.attr_name] = 1 - self.fdt_violation[i]
                else:
                    scores[parent.attr_name] = self.fdt_violation[i]
        return scores

    def test_cpt(self, row):
        if len(self.conditional_parents) > 0:
            parent_vals = []
            for p in self.conditional_parents:
                parent_vals.append(getattr(row, p.attr_name))
            if len(parent_vals) == 1:
                parent_vals = parent_vals[0]
            else:
                parent_vals = tuple(parent_vals)
            if parent_vals not in self.cpt:
                return self.new_relations
            val = getattr(row, self.attr_name)
            if val not in self.cpt[parent_vals]:
                return (1 - self.new_relations) * self.new_relations
            return (1 - self.new_relations) * self.cpt[parent_vals][val]
        return 1

    def test_value(self, row):
        if getattr(row, self.attr_name) not in self.values:
            return self.new_values
        else:
            return 1 - self.new_values
