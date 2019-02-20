import multiprocessing as mp
import re

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV

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
        # TODO: split per values of activities, if new activity -> use total set
        # set all values, independent of parents (used when new parent configuration has been found)
        vals = log[self.attr_name].values
        self.total_values = np.sort(vals)

        self.mean = np.mean(vals)
        self.std = np.std(vals)
        self.quantiles = (np.percentile(vals,25), np.percentile(vals,75))
        self.iqr = (self.quantiles[1] - self.quantiles[0])
        #self.iqr = 1.06 * np.std(vals) * np.power(len(vals), -1/5)
        #self.iqr = 2
        print("Quantiles:", self.quantiles)
        print("IQR:", self.iqr)

        #import matplotlib.pyplot as plt
        #plt.hist(vals)
        #plt.show()

        # Calculate best bandwith for KDE
        params = {'bandwidth': np.logspace(-2, 10, 100)}
        grid = GridSearchCV(KernelDensity(kernel='gaussian', rtol=1E-6), params, cv=2, n_jobs=mp.cpu_count(), verbose=3)
        grid.fit(log[[self.attr_name]].values)

        bw = grid.best_estimator_.bandwidth
        #bw = 20000

        print("Bandwidth:", bw)

        self.total_kernel = KernelDensity(kernel='gaussian', bandwidth=bw, rtol=1E-6).fit(log[[self.attr_name]].values)

        # set values per parent configuration
        self.iqrs = {}
        if len(self.discrete_parents) > 0:
            grouped = log.groupby([p.attr_name for p in self.discrete_parents])
            for group in grouped:
                if isinstance(group[0], int):
                    parent = str(group[0])
                else:
                    parent = "-".join([str(i) for i in group[0]])
                vals = group[1][self.attr_name].values
                self.values[parent] = np.sort(vals)
                self.hists[parent] = np.histogram(vals, bins=10)
                self.iqrs[parent] = np.percentile(vals,75) - np.percentile(vals,25)

                # Calculate best bandwith for KDE
                # params = {'bandwidth': np.logspace(-2, 1, 20)}
                # grid = GridSearchCV(KernelDensity(), params, cv=5)
                # grid.fit(group[1][[self.attr_name]].values)

                self.kernels[parent] = KernelDensity(kernel='gaussian', bandwidth=bw, rtol=1E-6).fit(group[1][[self.attr_name]].values)


        # Calculate best bandwith for KDE
        # params = {'bandwidth': np.logspace(-2, 1, 20)}
        # grid = GridSearchCV(KernelDensity(), params, cv=5)
        # grid.fit(log[[self.attr_name]].values)

        #print("Bandwidth:", grid.best_estimator_.bandwidth)
        #vals = log[self.attr_name].values
        #kernel = KernelDensity(kernel='gaussian', bandwidth=10).fit(log[[self.attr_name]].values)
        #x = np.linspace(min(vals) - 100, max(vals) + 100, 1000)
        #y = kernel.score_samples(x[:, np.newaxis])
        #import matplotlib.pyplot as plt
        #plt.title(self.attr_name)
        #plt.plot(x, y)
        #plt.show()

        self.hist, self.edges = np.histogram(self.total_values, bins=1000)

        def search_smallest_window(data, percentage):
            num_vals = int(len(data) * percentage)
            min_window = np.max(data)
            for i in range(len(data) - num_vals):
                window_size = data[i + num_vals] - data[i]
                if window_size < min_window:
                    min_window = window_size
            return min_window

        self.window_size = search_smallest_window(self.total_values, 0.75)
        print("WINDOW:", self.window_size)


    def train_continuous(self, log):
        if len(self.continuous_parents) > 0:
            parents = [p.attr_name for p in self.continuous_parents]
            vals = log[parents + [self.attr_name]].values

            # Calculate best bandwith for KDE
            params = {'bandwidth': np.logspace(-2, 1, 20)}
            grid = GridSearchCV(KernelDensity(), params, cv=2, n_jobs=mp.cpu_count())
            grid.fit(vals)


            self.kernel = KernelDensity(kernel='gaussian', bandwidth=grid.best_estimator_.bandwidth, rtol=1E-6).fit(vals) #grid.best_estimator_.bandwidth


    ###
    # Testing
    ###
    def test(self, row):
        score = self.test_value(row)
        score *= self.test_continuous(row)
        return score

    def test_value(self, row):
        val = getattr(row, self.attr_name)

        par = []
        for p in self.discrete_parents:
            par.append(str(getattr(row, p.attr_name)))
        par_val = "-".join(par)

        #2 Use Kernel Density scores for probability
        if par_val in self.kernels:
            return np.power(np.e, self.kernels[par_val].score_samples([[val]]))
        else:
            return np.power(np.e, self.total_kernel.score_samples([[val]]))

        #return np.power(np.e, self.total_kernel.score_samples([[val]])[0])


        """#1 Use CDF to estimate probability
        if par_val in self.values:
            val_list = self.values[par_val]
        else:
            val_list = self.total_values

        index = 0
        while (index < len(val_list)/2 and val_list[index] <= val) or (index < len(val_list) and val_list[index] < val):
            index += 1
        prob = index / len(val_list)
        if prob > 0.5:
            return (1 - prob) * 2 
        else:
            return prob * 2
        """

        """#3 Use window around data point and compare with total number of items
        window_size = self.window_size
        return len(self.total_values[(self.total_values >= (val - window_size)) & (self.total_values < (val + window_size))]) / len(self.total_values)
        """

        """#3a Use window around data point, restricted on activity
        if par_val in self.values:
            window_size = self.iqrs[par_val]
            return len(self.values[par_val][(self.values[par_val] >= (val - window_size)) & (
                        self.values[par_val] < (val + window_size))]) / len(self.values[par_val])
        else:
            window_size = self.iqr
            return len(self.total_values[(self.total_values >= (val - window_size)) & (
                        self.total_values < (val + window_size))]) / len(self.total_values)
        """

        """#4 Use Z-score
        z = np.abs((val - self.mean) / self.std)
        if z < 1:
            return 1
        else:
            return 0.5
        """

        """#5 Bin all values
        bin = 0
        for edge in self.edges[1:]:
            bin += 1
            if bin >= len(self.hist) or val < edge:
                break
        print(self.hist[bin - 1] / len(self.total_values))
        return self.hist[bin - 1] / len(self.total_values)
        """

        """#5a Bin all values per activity
        if par_val in self.hists:
            hist = self.hists[par_val][0]
            edge = self.hists[par_val][1]
            values = self.values[par_val]
        else:
            hist = self.hist
            edge = self.edges
            values = self.total_values
        bin = 0
        for e in edge:
            bin += 1
            if bin >= len(hist) or val < e:
                break
        return hist[bin - 1] / len(values)
        """

        """#6 Use Quartiles
        if val < self.outlier_vals[0] or val > self.outlier_vals[1]:
            return 0.1
        return 0.9
        """

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