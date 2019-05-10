import multiprocessing as mp
import re

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

import math
from scipy.spatial import ConvexHull

import Result

def calculate(trace):
    case = trace[0]
    data = trace[1]
    result = Result.Trace_result(case, data[time_attribute].iloc[0])
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


    def add_discrete_variable(self, name, new_values, empty_val):
        print("ADD Discrete:", name)
        self.variables[name] = Discrete_Variable(name, new_values, self.num_attrs, empty_val)
        m = re.search(r'Prev\d+$', name)
        if m is None:
            self.current_variables.append(name)

    def add_continuous_variable(self, name):
        print("ADD Continuous:", name)
        self.variables[name] = Continuous_Variable(name, self.num_attrs)
        m = re.search(r'Prev\d+$', name)
        if m is None:
            self.current_variables.append(name)

    def add_discretized_variable(self, name):
        print("ADD Discretized:", name)
        self.variables[name] = Discretized_Variable(name, self.num_attrs)
        m = re.search(r'Prev\d+$', name)
        if m is None:
            self.current_variables.append(name)

    def remove_variable(self, name):
        del self.variables[name]

    def iterate_variables(self):
        for key in self.variables:
            yield (key, self.variables[key])

    def iterate_current_variables(self):
        for key in self.current_variables:
            yield (key, self.variables[key])

    def get_variable(self, attr_name):
        return self.variables[attr_name]

    def train(self, data, single=False):
        if single:
            self.log = data
        else:
            self.log = data.contextdata

        for (_, value) in self.iterate_current_variables():
            value.train(self.log)
        print("Training Done")

    def train_durations(self):
        self.durations = {}
        groups = self.log.groupby(["Activity_Prev0", "Activity"])
        for group in groups:
            self.durations[group[0]] =  KernelDensity(kernel='gaussian', bandwidth=0.2, rtol=1E-2).fit(group[1]["duration_0"].values[:, np.newaxis])


    def calculate_scores_per_trace(self, data, accum_attr=None):
        """
        Return the result for all traces in the data
        """
        def initializer(init_model, time_attr):
            global model
            model = init_model
            global time_attribute
            time_attribute = time_attr

        data.create_k_context()

        print("EVALUATION: calculate scores")
        if accum_attr is None:
            accum_attr = self.trace_attr
        with mp.Pool(mp.cpu_count(), initializer, (self, data.time)) as p:
            scores = p.map(calculate, data.contextdata.groupby([accum_attr]))
        print("EVALUATION: Done")
        scores.sort(key=lambda l:l.time)
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

        with mp.Pool(mp.cpu_count()) as p:
            result = p.map(self.test_trace, log.groupby([self.trace_attr]))
        return result


    def test_parallel(self, data):
        njobs = mp.cpu_count()
        size = data.shape[1]
        if size < njobs:
            njobs = 1

        results = []
        chunks = np.array_split(data, njobs)

        for r in Parallel(n_jobs=njobs)(delayed(self.test)(d) for d in chunks):
            results.extend(r)
        return results

    def test_trace(self, trace):
        result = Result.Trace_result(trace[0])
        for row in trace[1].itertuples():
            result.add_event(self.test_row(row))
        return result

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
        result = Result.Event_result(row.Index)#, getattr(row, "anom_types"))
        for (key, value) in self.iterate_current_variables():
            result.set_attribute_score(value.attr_name, value.test(row))
        return result


class Variable:
    def __init__(self, attr_name, new_values, num_attrs, empty_val):
        self.attr_name = attr_name
        self.new_values = new_values
        self.new_relations = 0
        self.num_attrs = num_attrs
        self.empty_val = empty_val

    def __repr__(self):
        return self.attr_name

    def add_parent(self, var):
        pass

    def add_mapping(self, var):
        pass

    def train(self, log):
        pass

    def test(self, row):
        pass


class Discrete_Variable(Variable):
    def __init__(self, attr_name, new_values, num_attrs, empty_val):
        self.attr_name = attr_name
        self.new_values = new_values
        self.new_relations = 0
        #self.num_attrs = num_attrs
        self.empty_val = empty_val

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


    ###
    # Training
    ###
    def train(self, log):
        print("Train", self.attr_name)
        self.train_variable(log)
        self.train_fdt(log)
        self.train_cpt(log)
        self.set_new_relation(log)

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


    ###
    # Testing
    ###
    def test(self, row):
        total_score = 1
        for score in self.test_fdt(row).values():
            total_score *= score
        total_score *= self.test_cpt(row)
        total_score *= self.test_value(row)
        return total_score

    def test_fdt(self, row):
        scores = {}
        if len(self.functional_parents) > 0:
            for i in range(len(self.functional_parents)):
                parent = self.functional_parents[i]
                if getattr(row, parent.attr_name) not in self.fdt[i]:
                    scores[parent.attr_name] = 1 - self.fdt_violation[i]
                    self.fdt[i][getattr(row, parent.attr_name)] = getattr(row, self.attr_name)
                    self.values.add(getattr(row, self.attr_name))
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



class Continuous_Variable(Variable):
    def __init__(self, attr_name, num_attrs):
        self.attr_name = attr_name
        self.new_relations = 0
        self.num_attrs = num_attrs

        self.total_values = None
        self.values = {}
        self.k_dists = {}
        self.lofs = {}
        self.window_size = 0
        self.mean = 0
        self.std = 0
        self.hists = {}
        self.quantiles = None
        self.iqr = 0

        self.discrete_parents = []
        self.continuous_parents = []
        self.k_dist = None

    def __repr__(self):
        return self.attr_name

    def add_parent(self, var):
        print("ADDING PARENT:", var)
        if isinstance(var, Continuous_Variable):
            self.continuous_parents.append(var)
        else:
            self.discrete_parents.append(var)

    def add_mapping(self, var):
        raise NotImplementedError()


    ###
    # Training
    ###
    def train(self, log):
        self.train_variable(log)
        self.train_continuous(log)

    def train_variable(self, log):
        print("TRAINING CONT VALUE")

        vals = log[self.attr_name].values
        #self.total_values = np.sort(vals)[int(len(vals) * 0.05):int(len(vals) * 0.95)]
        self.total_values = np.sort(vals)

        print("Total values:", len(self.total_values), "Min:", np.min(self.total_values), "Max:", np.max(self.total_values), "Unique:", len(np.unique(self.total_values)))

        self.k_dist = k_distance(1).fit(self.total_values)


        # set values per parent configuration
        if len(self.discrete_parents) > 0:
            grouped = log.groupby([p.attr_name for p in self.discrete_parents])
            print("NUM GROUPS:", len(grouped))
            i = 0
            for group in grouped:
                # How to determine if amount of data is sufficient ?
                if len(group[1]) <= 20:
                    i += 1
                    continue

                if isinstance(group[0], int):
                    parent = str(group[0])
                else:
                    parent = "-".join([str(i) for i in group[0]])
                vals = group[1][self.attr_name].values
                print(group[0], len(vals), np.min(vals), np.mean(vals), np.max(vals))
                self.values[parent] = np.sort(vals)

                self.k_dists[parent] = k_distance(1).fit(self.values[parent])
                i += 1

    def train_continuous(self, log):
        if len(self.continuous_parents) > 0:
            parents = [p.attr_name for p in self.continuous_parents]
            vals = log[parents + [self.attr_name]].values

            # Calculate best bandwith for KDE
            params = {'bandwidth': np.logspace(-2, 1, 20)}
            grid = GridSearchCV(KernelDensity(), params, cv=2, n_jobs=mp.cpu_count())
            grid.fit(vals)


            self.kernel = KernelDensity(kernel='gaussian', bandwidth=grid.best_estimator_.bandwidth).fit(vals) #grid.best_estimator_.bandwidth


    ###
    # Testing
    ###
    def test(self, row):
        score1 = self.test_value(row)
        score2 = self.test_continuous(row)
        return score1 * score2

    def test_value(self, row, side="left"):
        val = getattr(row, self.attr_name)

        par = []
        for p in self.discrete_parents:
            par.append(str(getattr(row, p.attr_name)))
        par_val = "-".join(par)

        # Use Kernel Density scores for probability
        if par_val in self.k_dists:
            score = self.k_dists[par_val].score_distance([val], side)[0]
            #if score == 0:
            #    print("Value:", val, "Neighbour:", self.k_dists[par_val].get_k_neighbour([val]), "Distances:", self.k_dists[par_val].distances )
            return score
        else:
            return self.k_dist.score_distance([val], side)[0]


    def test_continuous(self, row):
        if len(self.continuous_parents) == 0:
            return 1

        parent_vals = np.zeros(len(self.continuous_parents) + 1)
        i = 0
        for p in self.continuous_parents:
            parent_vals[i] = getattr(row, p.attr_name)
            i += 1
        parent_vals[-1] = getattr(row, self.attr_name)
        parent_vals = parent_vals.reshape(1, -1)
        return np.power(np.e, self.kernel.score_samples(parent_vals)[0])

    def compare_left_right(self, row, score):
        val = getattr(row, self.attr_name)

        par = []
        for p in self.discrete_parents:
            par.append(str(getattr(row, p.attr_name)))
        par_val = "-".join(par)

        k_dist = None
        if par_val in self.k_dists:
            k_dist = self.k_dists[par_val]
        else:
            k_dist = self.k_dist

        print("Diff:", score, "Val", val, "Neighbour:", k_dist.get_k_neighbour([val]), k_dist.distances)


class k_distance(BaseEstimator):

    def __init__(self, k=5):
        self.k = k

    def fit(self, X):
        self.values = sorted(X)
        self.distances = sorted(self.get_k_neighbour(self.values))

        return self

    def score_distance(self, Y, side):
        neighbours = self.get_k_neighbour(Y)
        indexes = np.searchsorted(self.distances, neighbours, side="left")
        return 1 - (indexes / len(self.distances))

    def score(self):
        return np.sum(np.log(self.score_distance(self.values)))

    def get_k_neighbour(self, Y):
        dist = []
        indexes = np.searchsorted(self.values, Y)
        for i in range(len(indexes)):
            begin = max(0, indexes[i] - self.k - 1)
            end = min(len(self.values), indexes[i] + self.k + 1)
            if indexes[i] < len(self.values) and self.values[indexes[i]] == Y[i]:
                neighbours = self.values[begin:indexes[i]] + self.values[indexes[i] + 1:end]
            else:
                neighbours = self.values[begin:end]
            dist.append(sorted(np.abs(np.asarray(neighbours) - Y[i]))[min(self.k - 1, len(neighbours) - 1)])
        return dist



if __name__ == "__main__":
    train = [0,0,0,0,1,1,1,2,2,3,6,8,13,16,88]
    k_dist = k_distance(2).fit(train)

    test = [1, 4, 15, 50, 500]
    print(k_dist.score_distance(test))

    print(k_dist.distances)

class Discretized_Variable(Variable):
    def __init__(self, attr_name, num_attrs):
        self.attr_name = attr_name
        self.new_relations = 0
        self.num_attrs = num_attrs

    def __repr__(self):
        return self.attr_name

    def add_parent(self, var):
        pass

    def add_mapping(self, var):
        pass

    def train(self, log):
        self.value_counts = {}
        vc = log[self.attr_name].value_counts(normalize=True)
        print(vc)
        for row in vc.index:
            self.value_counts[row] = vc[row]

    def test(self, row):
        if getattr(row, self.attr_name) in self.value_counts:
            return self.value_counts[getattr(row, self.attr_name)]
        return 0
