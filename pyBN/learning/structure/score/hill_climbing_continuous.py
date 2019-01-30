"""
**********************
Greedy Hill-Climbing
for Structure Learning
**********************

Code for Searching through the space of
possible Bayesian Network structures.

Various optimization procedures are employed,
from greedy search to simulated annealing, and 
so on - mostly using scipy.optimize.

Local search - possible moves:
- Add edge
- Delete edge
- Invert edge

Strategies to improve Greedy Hill-Climbing:
- Random Restarts
    - when we get stuck, take some number of
    random steps and then start climbing again.
- Tabu List
    - keep a list of the K steps most recently taken,
    and say that the search cannt reverse (undo) any
    of these steps.
"""

import math
from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue

# from scipy.optimize import *
import numpy as np

from pyBN.classes.bayesnet import BayesNet
from pyBN.utils.graph import would_cause_cycle
from pyBN.utils.independence_tests import mutual_information

from sklearn.neighbors.kde import KernelDensity

import time

import matplotlib.pyplot as plt


# from heapq import *


def bay_net_size(bn):
    complexity = 0
    for node in bn.nodes():
        complexity += bn.F[node]["qi"]
    return complexity

def model_complexity(bn, nrows, metric="AIC"):
    if metric == "LL":
        return 0
    elif metric == "AIC":
        return bay_net_size(bn)
    elif metric == "BIC":
        return 0.5 * math.log(nrows) * bay_net_size(bn)

def model_score(data, bn):
    """
    Calculate the Log-likelihood score for the model given the data
    :param data:   the data
    :param bn:     the model
    :return:        returns the
    """
    total_score = 0
    num_rows = data.shape[0]

    for node in bn.nodes():
        print("Calculating for node", node)
        # Create all possible configurations of the parents
        parents = bn.parents(node)
        parent_configs = {}
        if len(parents) == 0:
            # Get the frequency for all occurring values
            vals = data[node].values[:, np.newaxis]
            print(vals)
            kdens = KernelDensity(kernel='gaussian', bandwidth=0.2, rtol=1E-2).fit(vals)
            total_score += kdens.score(vals)
        else: # TODO: change to CONTINUOUS
            # Create dataframe with only parents of node and convert to string
            str_data = data.values[:, [data.columns.get_loc(p) for p in parents]].astype('str')
            # Iterate over all rows and add row number to dict-entry of parent-values
            for row in range(num_rows):
                value = '-'.join(str_data[row, 0:len(parents)])
                if value not in parent_configs:
                    parent_configs[value] = []
                parent_configs[value].append(row)

            for parent_config in parent_configs:
                # Get the occurring values of this node for the current parent configuration
                filtered_data = data.values[parent_configs[parent_config], data.columns.get_loc(node)]
                # Get the frequencies of the occurring values
                freqs = np.bincount(filtered_data)
                for freq in freqs:
                    if freq > 0:
                        total_score += freq * math.log(freq / len(parent_configs[parent_config]))
            bn.F[node]["qi"] = data.drop_duplicates(list(parents)).shape[0]
    return total_score


class hill_climbing:

    def __init__(self, data, nodes):
        self.data = data
        self.nodes = nodes

        self.nrow = len(self.data)
        self.ncol = len(self.nodes)
        self.names = range(self.ncol)

        self.bandwidth = 0.05
        self.rtol = 1E-6

        # From Density Estimation for Statistics and Data Analysis, Bernard. W. Silverman, CRC ,1986
        #   (chapter Required sample size for given accuracy)
        self.sample_size = [4,19,67,223,768,2790,10700,43700,187000,842000]


    def hc(self, metric='AIC', max_iter=100, debug=False, restriction=None, whitelist=None):
        """
        Greedy Hill Climbing search proceeds by choosing the move
        which maximizes the increase in fitness of the
        network at the current step. It continues until
        it reaches a point where there does not exist any
        feasible single move that increases the network fitness.

        It is called "greedy" because it simply does what is
        best at the current iteration only, and thus does not
        look ahead to what may be better later on in the search.

        For computational saving, a Priority Queue (python's heapq)
        can be used	to maintain the best operators and reduce the
        complexity of picking the best operator from O(n^2) to O(nlogn).
        This works by maintaining the heapq of operators sorted by their
        delta score, and each time a move is made, we only have to recompute
        the O(n) delta-scores which were affected by the move. The rest of
        the operator delta-scores are not affected.

        For additional computational efficiency, we can cache the
        sufficient statistics for various families of distributions -
        therefore, computing the mutual information for a given family
        only needs to happen once.

        The possible moves are the following:
            - add edge
            - delete edge
            - invert edge

        Arguments
        ---------
        *data* : a nested numpy array
            The data from which the Bayesian network
            structure will be learned.

        *metric* : a string
            Which score metric to use.
            Options:
                - AIC
                - BIC / MDL
                - LL (log-likelihood)

        *max_iter* : an integer
            The maximum number of iterations of the
            hill-climbing algorithm to run. Note that
            the algorithm will terminate on its own if no
            improvement is made in a given iteration.

        *debug* : boolean
            Whether to print the scores/moves of the
            algorithm as its happening.

        *restriction* : a list of 2-tuples
            For MMHC algorithm, the list of allowable edge additions.

        Returns
        -------
        *bn* : a BayesNet object

        """

        # INITIALIZE NETWORK W/ NO EDGES
        # maintain children and parents dict for fast lookups
        self.c_dict = dict([(n,[]) for n in self.nodes])
        self.p_dict = dict([(n,[]) for n in self.nodes])

        self.restriction = restriction
        self.whitelist = whitelist

        if whitelist is None:
            self.whitelist = []
        for (u,v) in self.whitelist:
            if u in self.c_dict:
                self.c_dict[u].append(v)
            if v in self.p_dict:
                self.p_dict[v].append(u)
        print("Whitelist", self.whitelist)

        self.bn = BayesNet(self.c_dict)

        # COMPUTE INITIAL LIKELIHOOD SCORE
        print("Nodes:", list(self.bn.nodes()))

        # We do not take the complexity into account for Continuous Variables
        score = model_score(self.data, self.bn)# - model_complexity(self.bn, self.nrow, metric)
        print("Initial Score:", score)

        # CREATE EMPIRICAL DISTRIBUTION OBJECT FOR CACHING
        #ED = EmpiricalDistribution(data,names)

        _iter = 0
        improvement = True

        man = Manager()

        mut_inf_cache = man.dict()
        configs_cache = man.dict()

        x = []
        y = []

        while improvement:
            x.append(_iter)
            y.append(score)
            start_t = time.time()
            improvement = False
            max_delta = 0
            max_operation = None

            if debug:
                print('ITERATION: ' , _iter)


            return_queue = Queue()
            p_add = Process(target=self.test_arc_additions, args=(configs_cache, mut_inf_cache, return_queue))
            p_rem = Process(target=self.test_arc_deletions, args=(configs_cache, mut_inf_cache, return_queue))
            #p_rev = Process(target=self.test_arc_reversals, args=(configs_cache, mut_inf_cache, return_queue))

            p_add.start()
            p_rem.start()
            #p_rev.start()

            p_add.join()
            p_rem.join()
            #p_rev.join()

            while not return_queue.empty():
                results = return_queue.get()
                if results[1] > max_delta:
                    max_arc = results[0]
                    max_delta = results[1]
                    max_operation = results[2]

            ### DETERMINE IF/WHERE IMPROVEMENT WAS MADE ###
            if max_operation:
                score += max_delta
                improvement = True
                u,v = max_arc
                str_arc = [e for e in max_arc]
                if max_operation == 'Addition':
                    if debug:
                        print("delta:", max_delta)
                        print('ADDING: ' , str_arc , '\n')
                    self.p_dict[v].append(u)
                    self.bn.add_edge(u,v)
                elif max_operation == 'Deletion':
                    if debug:
                        print("delta:", max_delta)
                        print('DELETING: ' , str_arc , '\n')
                    self.p_dict[v].remove(u)
                    self.bn.remove_edge(u,v)
                elif max_operation == 'Reversal':
                    if debug:
                        print("delta:", max_delta)
                        print('REVERSING: ' , str_arc, '\n')
                    self.p_dict[v].remove(u)
                    self.bn.remove_edge(u,v)
                    self.p_dict[u].append(v)
                    self.bn.add_edge(v,u)
                print("Model score:", score)  # TODO: improve so only changed elements get an update
            else:
                if debug:
                    print('No Improvement on Iter: ' , _iter)
            print("Time for iteration:", time.time() - start_t)

            ### TEST FOR MAX ITERATION ###
            _iter += 1
        #    if _iter > max_iter:
        #        if debug:
        #            print('Max Iteration Reached')
        #        break


        bn = BayesNet(self.c_dict)
        print("Size of Cache", len(mut_inf_cache))
        print("SCORE =", score)

        plt.plot(x,y)
        plt.show()

        return bn

    def test_arc_reversals(self, configs_cache, mut_inf_cache, return_queue):
        print("Test Reversals")
        ### TEST ARC REVERSALS ###
        max_delta = 0
        max_operation = None
        max_arc = None
        max_qi = 0
        for u in self.bn.nodes():
            for v in self.c_dict[u]:
                if not would_cause_cycle(self.c_dict, v, u, reverse=True) and (
                        self.restriction is None or (v, u) in self.restriction): # and (
 #                       self.whitelist is None or (u,v) not in self.whitelist):
                    # SCORE FOR 'U' -> gaining 'v' as parent
                    old_cols = (u,) + tuple(self.p_dict[u])  # without 'v' as parent
                    if old_cols not in mut_inf_cache:
                        mut_inf_cache[old_cols] = mutual_information(self.data[list(old_cols)])
                    mi_old = mut_inf_cache[old_cols]

                    new_cols = old_cols + (v,)  # with 'v' as parent
                    if new_cols not in mut_inf_cache:
                        mut_inf_cache[new_cols] = mutual_information(self.data[list(new_cols)])
                    mi_new = mut_inf_cache[new_cols]

                    delta1 = self.nrow * (mi_new - mi_old)  # Add difference in complexity -> recalculate qi for node v

                    # SCORE FOR 'V' -> losing 'u' as parent
                    old_cols = (v,) + tuple(self.p_dict[v])  # with 'u' as parent
                    if old_cols not in mut_inf_cache:
                        mut_inf_cache[old_cols] = mutual_information(self.data[list(old_cols)])
                    mi_old = mut_inf_cache[old_cols]

                    new_cols = tuple([i for i in old_cols if i != u])  # without 'u' as parent
                    if new_cols not in mut_inf_cache:
                        mut_inf_cache[new_cols] = mutual_information(self.data[list(new_cols)])
                    mi_new = mut_inf_cache[new_cols]

                    delta2 = self.nrow * (mi_new - mi_old)  # Add difference in complexity -> recalculate qi for node v

                    # COMBINED DELTA-SCORES
                    ri1 = self.bn.F[u]['ri']
                    qi1 = self.bn.F[u]['qi']
                    qi_new1 = calc_num_parent_configs(self.data, self.bn.parents(u) + [v], configs_cache)
                    ri2 = self.bn.F[v]['ri']
                    qi2 = self.bn.F[v]['qi']
                    qi_new2 = calc_num_parent_configs(self.data, [x for x in self.bn.parents(v) if x != u],
                                                      configs_cache)

                    delta_score = delta1 + delta2 - (ri2 * (qi_new2 - qi2) - (qi_new2 - qi2)) - (
                                ri1 * (qi_new1 - qi1) - (
                                    qi_new1 - qi1))  # Add difference in complexity -> recalculate qi for node u and v

                    if delta_score - max_delta > 10 ** (-10):
                        max_delta = delta_score
                        max_operation = 'Reversal'
                        max_arc = (u, v)
                        max_qi = (qi_new1, qi_new2)
        return_queue.put((max_arc, max_delta, max_operation, max_qi))

    def test_arc_deletions(self, configs_cache, l_inf_cache, return_queue):
        print("Test Deletions")
        ### TEST ARC DELETIONS ###
        max_delta = 0
        max_operation = None
        max_arc = None
        max_qi = 0
        for u in self.bn.nodes():
            for v in [n for n in self.c_dict[u] if (u,n) not in self.whitelist]:
                #if (u,v) not in self.whitelist:
                    # SCORE FOR 'V' -> losing a parent
                    old_cols = (v,) + tuple(self.p_dict[v])  # with 'u' as parent
                    if old_cols not in l_inf_cache:
                        vals = self.data[list(old_cols)].values
                        kdens = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth, rtol=self.rtol).fit(vals)
                        l_inf_cache[old_cols] = kdens.score(vals)
                    old_cols2 = tuple(self.p_dict[v])
                    if old_cols2 not in l_inf_cache:
                        vals = self.data[list(old_cols2)].values
                        kdens = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth, rtol=self.rtol).fit(vals)
                        l_inf_cache[old_cols2] = kdens.score(vals)
                    l_old = l_inf_cache[old_cols] - l_inf_cache[old_cols2]

                    new_cols = tuple([i for i in old_cols if i != u])  # without 'u' as parent
                    if len(new_cols) == 1:
                        if new_cols not in l_inf_cache:
                            vals = self.data[list(new_cols)].values
                            kdens = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth, rtol=self.rtol).fit(vals)
                            l_inf_cache[new_cols] = kdens.score(vals)
                        l_new = l_inf_cache[new_cols]
                    else:
                        if new_cols not in l_inf_cache:
                            vals = self.data[list(new_cols)].values
                            kdens = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth, rtol=self.rtol).fit(vals)
                            l_inf_cache[new_cols] = kdens.score(vals)
                        if tuple([n for n in self.p_dict[v] if n != u]) not in l_inf_cache:
                            vals = self.data[list(self.p_dict[v])].values
                            kdens = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth, rtol=self.rtol).fit(vals)
                            l_inf_cache[tuple(self.p_dict[v])] = kdens.score(vals)
                        l_new = l_inf_cache[old_cols] - l_inf_cache[tuple(self.p_dict[v])]

                    delta_score = (l_new - l_old) #- self.sample_size[min(len(new_cols), len(self.sample_size))]

                    if delta_score - max_delta > 10 ** (-10):
                        max_delta = delta_score
                        max_operation = 'Deletion'
                        max_arc = (u, v)
        return_queue.put((max_arc, max_delta, max_operation, max_qi))

    def test_arc_additions(self, configs_cache, l_inf_cache, return_queue):
        print("Test Additions")
        ### TEST ARC ADDITIONS ###
        max_delta = 0
        max_operation = None
        max_arc = None
        procs = []
        result_queue = Queue()
        for u in self.bn.nodes():
            p = Process(target=self.test_arcs, args=(configs_cache, l_inf_cache, u, result_queue))
            procs.append(p)
            p.start()

        for p in procs:
            p.join()

        while not result_queue.empty():
            results = result_queue.get()

            if results[1] - max_delta > 10 ** (-10):
                max_arc = results[0]
                max_delta = results[1]
                max_operation = results[2]
        return_queue.put((max_arc, max_delta, max_operation))

    def test_arcs(self, configs_cache, l_inf_cache, u, result_queue):
        max_delta = 0
        max_operation = None
        max_arc = None
        for v in [n for n in self.bn.nodes() if u != n and n not in self.c_dict[u] and not would_cause_cycle(self.c_dict, u, n)]:
            # FOR MMHC ALGORITHM -> Edge Restrictions
            if self.restriction is None or (u, v) in self.restriction:
                # SCORE FOR 'V' -> gaining a parent
                old_cols = (v,) + tuple(self.p_dict[v])  # without 'u' as parent
                if len(old_cols) == 1:
                    if old_cols not in l_inf_cache:
                        vals = self.data[list(old_cols)].values
                        kdens = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth, rtol=self.rtol).fit(vals)
                        l_inf_cache[old_cols] = kdens.score(vals)
                    l_old = l_inf_cache[old_cols]
                else:
                    if old_cols not in l_inf_cache:
                        vals = self.data[list(old_cols)].values
                        kdens = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth, rtol=self.rtol).fit(vals)
                        l_inf_cache[old_cols] = kdens.score(vals)
                    if tuple(self.p_dict[v]) not in l_inf_cache:
                        vals = self.data[list(self.p_dict[v])].values
                        kdens = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth, rtol=self.rtol).fit(vals)
                        l_inf_cache[tuple(self.p_dict[v])] = kdens.score(vals)
                    l_old = l_inf_cache[old_cols] - l_inf_cache[tuple(self.p_dict[v])]

                new_cols = old_cols + (u,)  # with'u' as parent
                if new_cols not in l_inf_cache:
                    vals = self.data[list(new_cols)].values
                    kdens = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth, rtol=self.rtol).fit(vals)
                    l_inf_cache[new_cols] = kdens.score(vals)
                new_cols2 = tuple(self.p_dict[v]) + (u,)
                if new_cols2 not in l_inf_cache:
                    vals = self.data[list(new_cols2)].values
                    kdens = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth, rtol=self.rtol).fit(vals)
                    l_inf_cache[new_cols2] = kdens.score(vals)
                l_new = l_inf_cache[new_cols] - l_inf_cache[new_cols2]

                delta_score = (l_new - l_old) - self.sample_size[min(len(new_cols), len(self.sample_size))]

                if delta_score - max_delta > 10 ** (-10):
                    max_delta = delta_score
                    max_operation = 'Addition'
                    max_arc = (u, v)
        result_queue.put((max_arc, max_delta, max_operation))


def learn_continuous_net(train):
    hc = hill_climbing(train, train.columns)
    net = hc.hc(debug=True)

    from extended_Dynamic_Bayesian_Network import extendedDynamicBayesianNetwork

    edbn = extendedDynamicBayesianNetwork(len(train.columns), 0, None)
    for col in train.columns:
        edbn.add_variable(col, 1, None)

    print("FOUND RELATIONS:")
    for edge in net.edges():
        print(edge)
        edbn.add_continuous_parent(edge[0], edge[1])

    edbn.train(train, single=True)
    return edbn

def score_continuous_net(model, test, label_attr):
    import Utils.PlotResults as plot

    ranking = model.test(test)
    ranking.sort(key=lambda l: l[0].get_total_score())
    scores = []
    y = []
    for r in ranking:
        scores.append((getattr(r[1], "Index"), r[0].get_total_score(), getattr(r[1], label_attr) != 0))
        y.append(r[0].get_total_score())
    print(len(scores))

    with open("../output.csv", "w") as fout:
        for s in scores:
            fout.write(",".join([str(i) for i in s]))
            fout.write("\n")

    plot.plot_single_roc_curve("../output.csv")
    plot.plot_single_prec_recall_curve("../output.csv")


if __name__ == "__main__":
    import pandas as pd
    import Utils.PlotResults as plot

    log = pd.read_csv("../../../../Data/creditcard.csv", nrows=10000, dtype='float64').drop(columns=["Time"])

    train = log[:1000]
    test = log[1000:]

    print(log["Class"].value_counts())
    print(train["Class"].value_counts())
    print(test["Class"].value_counts())

    train = train[train.Class == 0] # Only keep non-anomalies
    train = train.drop(columns=["Class"]) # Drop Class label
    train = train.drop(columns=["Amount"])

    model = learn_continuous_net(train)
    score_continuous_net(model, test)



    #plt.plot(list(range(len(y))), y)
    #plt.show()
