import multiprocessing as mp
import re

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors.kde import KernelDensity

import Result

def calculate(trace):
    case = trace[0]
    data = trace[1]
    result = Result.Trace_result(case)
    for row in data.itertuples():
        e_result = model.test_row(row)
        result.add_event(e_result)
    return result

class extendedDynamicBayesianNetwork():
    """
    Class for representing an extended Dynamic Bayesian Network (eDBN)
    """

    def __init__(self, num_attrs, k, trace_attr):
        self.variables = {}
        self.current_variables = []
        self.num_attrs = num_attrs
        self.log = None
        self.k = k
        self.trace_attr = trace_attr
        self.durations = None

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

    def train(self, data):
        self.log = data.contextdata

        for (_, value) in self.iterate_current_variables():
            self.train_var(value)
        print("Training Done")

    def train_var(self, var):
        print("Training", var.attr_name)
        var.train_variable(self.log)
        var.train_fdt(self.log)
        var.train_cpt(self.log)
        var.set_new_relation(self.log)
        return var

    def train_durations(self):
        self.durations = {}
        groups = self.log.groupby(["Activity_Prev0", "Activity"])
        for group in groups:
            self.durations[group[0]] =  KernelDensity(kernel='gaussian', bandwidth=0.2, rtol=1E-2).fit(group[1]["duration_0"].values[:, np.newaxis])

    def calculate_scores_per_trace(self, data, accum_attr=None):
        """
        Return the result for all traces in the data
        """
        def initializer(init_model):
            global model
            model = init_model

        data.create_k_context()
        data = data.contextdata

        print("EVALUATION: calculate scores")
        if accum_attr is None:
            accum_attr = self.trace_attr
        with mp.Pool(mp.cpu_count(), initializer, (self,)) as p:
            scores = p.map(calculate, data.groupby([accum_attr]))
        print("EVALUATION: Done")

        return scores


    def calculate_scores_per_attribute(self, data, accum_attr = None):
        """
        Return the results for all traces per attribute
        """
        result = self.calculate_scores_per_trace(data, accum_attr)
        scores = {}

        print("EVALUATION: Combine by attribute")
        for attribute in [x for x in self.current_variables]:
            for trace_result in result:
                if attribute not in scores:
                    scores[attribute] = []
                scores[attribute].append(trace_result.get_attribute_score(attribute))

        return scores

    def test_data(self, data):
        """
        Compute the score for all events in the k-context of the data
        """
        print("EVALUATION: calculate scores")
        data.create_k_context()
        log = data.contextdata

        njobs = mp.cpu_count()
        size = len(log)
        if size < njobs:
            njobs = 1

        results = []
        chunks = np.array_split(log, njobs)

        for r in Parallel(n_jobs=njobs)(delayed(self.test)(d) for d in chunks):
            results.extend(r)
        results.sort(key=lambda l: l[0].get_total_score())
        return results

    def test(self, data):
        """
        Compute the scores for all events in the data
        :param data:
        :return:
        """
        ranking = []
        for row in data.itertuples():
            ranking.append((self.test_row(row), row))
        return ranking

    def test_row(self, row):
        """
        Return the score for the k-context of a single event
        """
        result = Result.Event_result(row.Index)
        for (key, value) in self.iterate_current_variables():
            score_fdt = 1
            fdt_scores = value.test_fdt(row)
            for fdt_score in fdt_scores:
                score_fdt *= fdt_scores[fdt_score]
            result.set_attribute_score(value.attr_name, score_fdt * value.test_cpt(row) * value.test_value(row))
        return result


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
