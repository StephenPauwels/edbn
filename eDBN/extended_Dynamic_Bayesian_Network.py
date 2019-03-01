import multiprocessing as mp
import re

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

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


        """
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
        """

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
        result = Result.Event_result(row.Index)
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
        self.num_attrs = num_attrs
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
        self.kernels = {}
        self.window_size = 0
        self.mean = 0
        self.std = 0
        self.hists = {}
        self.quantiles = None
        self.iqr = 0

        self.discrete_parents = []
        self.continuous_parents = []
        self.kernel = None

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
        self.total_values = np.sort(vals)

        noise = 1 / (0.0001 + self.total_values[-1] - self.total_values[0])
        print("Noise:",  noise)

        # Calculate best bandwith for KDE
        #params = {'bandwidth': np.logspace(-2, 10, 25), 'alpha': np.linspace(0,1,11)}


        # Tophat kernel
        m = []
        for i in range(len(self.total_values) - 1):
            m.append(self.total_values[i+1] - self.total_values[i])
        bw = max(0.0001, np.mean(m))
        
        params = {'alpha': np.linspace(0,1,11)}
        grid = GridSearchCV(tophat_KDE(noise=noise, bandwidth=bw), params, cv=2, n_jobs=mp.cpu_count(), verbose=0)
        grid.fit(self.total_values)

        #bw = grid.best_estimator_.bandwidth

        a = grid.best_estimator_.alpha

        print("Bandwidth:", bw, "Alpha:", a)
        self.total_kernel = tophat_KDE(bandwidth=bw, noise=noise, alpha=a).fit(self.total_values)
        ###
        """

        ## Gaussian Kernel
        params = {'bandwidth': np.logspace(-2, 10, 25)}
        grid = GridSearchCV(KernelDensity(kernel='gaussian', rtol=1E-6), params, cv=2, n_jobs=mp.cpu_count(), verbose=1)
        grid.fit(log[[self.attr_name]].values)

        #bw = grid.best_estimator_.bandwidth
        bw = 100000

        print("Bandwidth:", bw)
        self.total_kernel = KernelDensity(kernel='gaussian', bandwidth=bw, rtol=1E-6).fit(log[[self.attr_name]].values)
        ###
        """
        # set values per parent configuration
        if len(self.discrete_parents) > 0:
            grouped = log.groupby([p.attr_name for p in self.discrete_parents])
            print("NUM GROUPS:", len(grouped))
            i = 0
            for group in grouped:
                if len(group[1]) < 2:
                    i += 1
                    continue

                if isinstance(group[0], int):
                    parent = str(group[0])
                else:
                    parent = "-".join([str(i) for i in group[0]])
                vals = group[1][self.attr_name].values
                self.values[parent] = np.sort(vals)

                self.kernels[parent] = tophat_KDE(bandwidth=bw, noise=noise, alpha=a).fit(self.values[parent])
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

    def test_value(self, row):
        val = getattr(row, self.attr_name)

        par = []
        for p in self.discrete_parents:
            par.append(str(getattr(row, p.attr_name)))
        par_val = "-".join(par)

        # Use Kernel Density scores for probability
        if par_val in self.kernels:
            #return np.power(np.e, self.kernels[par_val].score_samples([[val]])[0])
            return self.kernels[par_val].score_samples([[val]])[0]
        else:
            #return np.power(np.e, self.total_kernel.score_samples([[val]])[0])
            return self.total_kernel.score_samples([[val]])[0]



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


class tophat_KDE(BaseEstimator):

    def __init__(self, bandwidth=1, noise=0, alpha=0):
        self.bandwidth = bandwidth
        self.noise = noise
        self.alpha = alpha

    def fit(self, X):
        self.values = sorted(X)
        self.distr = (1-self.alpha) / (self.bandwidth * 2 * len(self.values))
        return self

    def score(self, X):
        return np.sum(np.log(self.score_samples(X)))

    def score_samples(self, X):
        X = np.asarray(X)
        begin_indexes = np.searchsorted(self.values, X - self.bandwidth, side='left')
        end_indexes = np.searchsorted(self.values, X + self.bandwidth, side='right')
        score = self.alpha * self.noise + (end_indexes - begin_indexes) * self.distr
        if len(score) == 1:
            score = score[0]
        return score
