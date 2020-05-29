"""
    Author: Stephen Pauwels
"""

import math
import multiprocessing as mp
from multiprocessing import Manager, Process, Queue

import numpy as np
import sklearn.metrics as skm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


class Structure_learner():

    def __init__(self, log, nodes):
        self.log = log
        self.data = self.log.get_data()
        self.nrow = len(self.data)

        self.nodes = dict([(n,{}) for n in nodes])

    def model_complexity(self):
        """
        Calculate the complexity factor of the current model
        """
        total_complexity = 0

        for node in self.nodes.keys():
            if self.log.isNumericAttribute(node):
                total_complexity += self.numericalComplexity(1, 1)
            elif self.log.isCategoricalAttribute(node):
                total_complexity += self.categoricalComplexity(node)
        return total_complexity

    def categoricalComplexity(self, node):
        """
        Calculate the complexity factor of a particular categorical node
        """
        return self.nodes[node]["qi"]

    def numericalComplexity(self, num_columns, num_disc_parents):
        """
        Calculate the complexity factor of a particular numerical node
        """
        complexity = [4,19,67,223,768,2790,10700,43700,187000,842000]
        return num_disc_parents * complexity[min(num_columns, len(complexity) - 1)]

    def model_score(self, cache, bandwidth_cache):
        """
        Calculate the score for the entire model
        """
        total_score = 0

        for node in self.nodes.keys():
            if self.log.isNumericAttribute(node):
                total_score += self.numericalScore(node, cache, bandwidth_cache)
            elif self.log.isCategoricalAttribute(node):
                s, q = self.categoricalScore(node)
                self.nodes[node]["qi"] = q
                total_score += s
        return total_score

    def categoricalScore(self, node, use_parents=None):
        """
        Calculate the score for a particular categorical node
        """
        total_score = 0
        qi = 0
        num_rows = self.data.shape[0]

        # Create all possible configurations of the parents
        parents = self.p_dict[node]
        if use_parents:
            parents = use_parents

        parent_configs = {}

        # Using bincount is faster than numpy.unique
        values = self.data.values[:, self.data.columns.get_loc(node)].astype(int)
        self.nodes[node]["ri"] = len([x for x in np.bincount(values) if x > 0])

        if len(parents) == 0:
            # Get the frequency for all occurring values
            freqs = np.bincount(values)
            for count in freqs:
                if count != 0:
                    total_score += count * math.log(count / num_rows)
            qi = 1
        else:
            # Create dataframe with only parents of node and convert to string
            str_data = self.data.values[:, [self.data.columns.get_loc(p) for p in parents]].astype('str')
            # Iterate over all rows and add row number to dict-entry of parent-values
            for row in range(num_rows):
                value = '-'.join(str_data[row, 0:len(parents)])
                if value not in parent_configs:
                    parent_configs[value] = []
                parent_configs[value].append(row)

            for parent_config in parent_configs:
                # Get the occurring values of this node for the current parent configuration
                filtered_data = self.data.values[parent_configs[parent_config], self.data.columns.get_loc(node)].astype(int)
                # Get the frequencies of the occurring values
                freqs = np.bincount(filtered_data)
                for freq in freqs:
                    if freq > 0:
                        total_score += freq * math.log(freq / len(parent_configs[parent_config]))
            qi = self.data.drop_duplicates(list(parents)).shape[0]
        return total_score, qi

    def numericalScore(self, node, cache, bandwidth_cache):
        """
        Calculate the score for a particular numerical node
        """
        cache[(node,)] = self.calc_kde_score([node], [], bandwidth_cache, n_jobs=mp.cpu_count())
        return cache[(node,)]

    def categoricalDelta(self, node, parents, cache):
        """
        Score for a particular categorical node given a set of categorical parents
        """
        cols = (node,) + tuple(parents)
        if cols not in cache:
            cache[cols] = mutual_information(self.data[list(cols)])
        mi = cache[cols]
        qi = self.data.drop_duplicates(list(cols)).shape[0]

        return self.nrow * mi - qi, qi

    def numericalDelta(self, node, parents, cache, bandwidth_cache):
        """
        Score for a particular numerical node given a set of parents
        """
        cont_parents = []
        disc_parents = []

        for parent in parents:
            if self.log.isNumericAttribute(parent):
                cont_parents.append(parent)
            else:
                disc_parents.append(parent)

        cols1 = (node,) + tuple(cont_parents)
        if cols1 + tuple(disc_parents) not in cache:
            cache[cols1 + tuple(disc_parents)] = self.calc_kde_score(cols1, disc_parents, bandwidth_cache)

        if len(cols1) == 1:
            return cache[cols1 + tuple(disc_parents)] - self.numericalComplexity(1, len(disc_parents))
        else:
            cols2 = tuple(cont_parents)
            if cols2 + tuple(disc_parents) not in cache:
                cache[cols2 + tuple(disc_parents)] = self.calc_kde_score(cols2, disc_parents, bandwidth_cache)
            return (cache[cols1 + tuple(disc_parents)] - cache[cols2 + tuple(disc_parents)]) - self.numericalComplexity(len(cols2) + 1, len(disc_parents))


    def calc_kde_score(self, cols, disc_parents, bandwidth_cache, n_jobs = 1):
        partitions = []
        if len(disc_parents) > 0:
            partitions = self.data.groupby(disc_parents)
        else:
            partitions = [("", self.data)]

        score = 0

        to_check = []
        for partition in partitions:
            if len(partition[1]) > 20: # Only consider partitions that are large enough
                vals = partition[1][list(cols)].values
                vals_hash = hash(vals.tobytes())
                # Calculate best bandwith for KDE
                if vals_hash not in bandwidth_cache:
                    params = {'bandwidth': np.logspace(-2, 5, 20)}
                    grid = GridSearchCV(KernelDensity(kernel='gaussian', rtol=1E-6), params, cv=2, verbose=0, n_jobs=n_jobs, iid=False)
                    grid.fit(vals)
                    bandwidth_cache[vals_hash] = grid.best_estimator_.bandwidth
                score += KernelDensity(kernel='gaussian', bandwidth=bandwidth_cache[vals_hash], rtol=1E-6).fit(vals).score(vals)
            else:
                to_check.extend(partition[1][list(cols)].values)

        if len(to_check) > 0: # Check all partitions that are considered too small together
            vals = self.data[list(cols)].values
            vals_hash = hash(vals.tobytes())
            # Calculate best bandwith for KDE
            if vals_hash not in bandwidth_cache:
                params = {'bandwidth': np.logspace(-2, 5, 20)}
                grid = GridSearchCV(KernelDensity(kernel='gaussian', rtol=1E-6), params, cv=2, verbose=0, n_jobs=n_jobs, iid=False)
                grid.fit(vals)
                bandwidth_cache[vals_hash] = grid.best_estimator_.bandwidth
            score += KernelDensity(kernel='gaussian', bandwidth=bandwidth_cache[vals_hash], rtol=1E-6).fit(vals).score(to_check)

        return score


    def learn(self, restrictions=None, whitelist=None):
        # maintain children and parents dict for fast lookups
        self.c_dict = dict([(n,[]) for n in self.nodes])
        self.p_dict = dict([(n,[]) for n in self.nodes])

        self.restriction = restrictions

        if whitelist is None:
            whitelist = []
        for (u,v) in whitelist:
            if u in self.c_dict:
                self.c_dict[u].append(v)
            if v in self.p_dict:
                self.p_dict[v].append(u)
        self.whitelist = whitelist

        print("LEARN: Nodes:", self.nodes.keys())

        man = Manager()
        cache = man.dict()
        bandwidth_cache = man.dict()

        score = self.model_score(cache, bandwidth_cache) - self.model_complexity()
        print("LEARN: Initial Model Score:", score)

        _iter = 0
        improvement = True


        while improvement:
            print("LEARN: Iteration", _iter)
            improvement = False
            max_delta = 0
            max_operation = None

            return_queue = Queue()
            p_add = Process(target=self.test_arc_additions, args=(cache, bandwidth_cache, return_queue))
            p_rem = Process(target=self.test_arc_deletions, args=(cache, bandwidth_cache, return_queue))

            p_add.start()
            p_rem.start()

            p_add.join()
            p_rem.join()

            while not return_queue.empty():
                results = return_queue.get()
                if results[1] > max_delta:
                    max_arc = results[0]
                    max_delta = results[1]
                    max_operation = results[2]
                    max_qi = results[3]

            ### DETERMINE IF/WHERE IMPROVEMENT WAS MADE ###
            if max_operation:
                score += max_delta
                improvement = True
                u,v = max_arc
                str_arc = [e for e in max_arc]
                if max_operation == 'Addition':
                    self.p_dict[v].append(u)
                    self.c_dict[u].append(v)
                    if max_qi is not None:
                        self.nodes[v]["qi"] = max_qi
                    print("LEARN: Add:", u, "->", v)
                elif max_operation == 'Deletion':
                    self.p_dict[v].remove(u)
                    self.c_dict[u].remove(v)
                    if max_qi is not None:
                        self.nodes[v]["qi"] = max_qi
                    print("LEARN: Delete:", u, "->", v)
                print("LEARN: Delta:", max_delta)
                print("LEARN: Model score:", score)
            _iter += 1

        edges = []
        for node in self.nodes.keys():
            for child in self.c_dict[node]:
                edges.append((node, child))

        return edges


    def test_arc_deletions(self, cache, bandwidth_cache, return_queue):
        ### TEST ARC DELETIONS ###
        max_delta = 0
        max_operation = None
        max_arc = None
        max_qi = 0
        for u in self.nodes.keys():
            for v in [n for n in self.c_dict[u] if (u,n) not in self.whitelist]:
                if self.log.isCategoricalAttribute(v):
                    old_cache_cols = (v,) + tuple(self.p_dict[v])
                    if old_cache_cols not in cache:
                        cache[old_cache_cols] = self.categoricalScore(v, self.p_dict[v])
                    old_score, old_qi = cache[old_cache_cols]

                    new_cache_cols = (v,) + tuple([i for i in self.p_dict[v] if i != u])
                    if new_cache_cols not in cache:
                        cache[new_cache_cols] = self.categoricalScore(v, [i for i in self.p_dict[v] if i != u])
                    new_score, new_qi = cache[new_cache_cols]
                elif self.log.isNumericAttribute(v):
                    old_score = self.numericalDelta(v, [par for par in self.p_dict[v]], cache, bandwidth_cache)
                    new_score = self.numericalDelta(v, [par for par in self.p_dict[v]], cache, bandwidth_cache)
                    new_qi = None
                delta_score = (new_score - new_qi) - (old_score - old_qi)

                if delta_score - max_delta > 10 ** (-10):
                    max_delta = delta_score
                    max_operation = 'Deletion'
                    max_arc = (u, v)
                    max_qi = new_qi
        return_queue.put((max_arc, max_delta, max_operation, max_qi))

    def test_arc_additions(self, cache, bandwidth_cache, return_queue):
        # Create all tuples of edges to check
        edges = []
        for u in self.nodes.keys():
            for v in self.nodes.keys():
                if u != v and v not in self.c_dict[u] and not would_cause_cycle(self.c_dict, u, v):
                    if self.restriction is None or (u, v) in self.restriction:
                        edges.append((u, v))

        from functools import partial
        test_func = partial(self.test_arcs, cache=cache, bandwidth_cache=bandwidth_cache)
        with mp.Pool(mp.cpu_count()) as p:
            results = p.map(test_func, edges)

        max_result = max(results, key=lambda l: l[0])

        return_queue.put((max_result[2], max_result[0], max_result[1], max_result[3]))


    def test_arcs(self, edge, cache, bandwidth_cache):
        u = edge[0]
        v = edge[1]
        if self.log.isCategoricalAttribute(v):
            old_cache_cols = (v,) + tuple(self.p_dict[v])
            if old_cache_cols not in cache:
                cache[old_cache_cols] = self.categoricalScore(v, self.p_dict[v])
            old_score, old_qi = cache[old_cache_cols]

            new_cache_cols = (v,) + tuple(self.p_dict[v] + [u])
            if new_cache_cols not in cache:
                cache[new_cache_cols] = self.categoricalScore(v, self.p_dict[v] + [u])
            new_score, new_qi = cache[new_cache_cols]
        elif self.log.isNumericAttribute(v):
            old_score = self.numericalDelta(v, [par for par in self.p_dict[v]], cache, bandwidth_cache)
            new_score = self.numericalDelta(v, [par for par in self.p_dict[v] + [u]], cache, bandwidth_cache)
            new_qi = None
        delta_score = (new_score - new_qi) - (old_score - old_qi)
        return delta_score, "Addition", (u, v), new_qi


def would_cause_cycle(graph, u, v, visited = None):
    """
    Check if adding the edge (u,v) would create a cycle in the graph
    """
    if visited is None:
        visited = []

    if v in visited:
        return True

    for child in graph[v]:
        if would_cause_cycle(graph, v, child, visited + [u]):
            return True
    return False

def mutual_information(data):
    cols = len(data.columns)
    data = data.values
    if cols == 1:
        # TODO: Change -> first attr is a constant (empty parent)
        #return skm.mutual_info_score(data[:,0], data[:,0])
        return 0
    if cols == 2:
        return skm.mutual_info_score(data[:,0], data[:,1])
    elif cols > 2:
        data = data.astype('str')
        for i in range(len(data)):
            data[i,1] = data[i,1:cols].tostring()
        return skm.mutual_info_score(data[:,0].astype('str'), data[:,1])



def score_continuous_net(model, test, label_attr, output_file=None, title=None):
    import Utils.PlotResults as plot

    ranking = model.test_parallel(test)
    ranking.sort(key=lambda l: l[0].get_total_score())
    scores = []
    y = []
    for r in ranking:
        scores.append((getattr(r[1], "Index"), r[0].get_total_score(), getattr(r[1], label_attr) != 0))
        y.append(r[0].get_total_score())
    print(len(scores))

    if output_file is None:
        output_file = "../output.csv"

    with open(output_file, "w") as fout:
        for s in scores:
            fout.write(",".join([str(i) for i in s]))
            fout.write("\n")

    plot.plot_single_roc_curve(output_file, title)
    plot.plot_single_prec_recall_curve(output_file, title)

if __name__ == "__main__":
    vals = [1,1,1,1,2,4,5,5,7,8,10]
    input = []
    for v in vals:
        input.append([v])
    kdens = KernelDensity(kernel='gaussian', bandwidth=1, rtol=1E-6).fit(input)
    X = np.linspace(0,15,50)
    X = X.reshape(-1,1)
    Y = kdens.score_samples(X)
    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best")

    plt.plot(X,np.e**Y, label="KDE estimate", color=(31/255, 119/255, 180/255))
    plt.plot(vals, [0.01 for _ in vals], "x", label="Values", color=(148/255, 103/255, 189/255))
    bins = []
    for i in range(11):
        bins.append(i + 0.5)
    plt.hist(vals, density=True, label="Histogram", bins=bins, color=(255/255, 127/255, 14/255))
    plt.show()


