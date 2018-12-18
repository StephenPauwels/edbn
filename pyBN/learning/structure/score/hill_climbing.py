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

import time


# from heapq import *


def bay_net_size(bn):
    complexity = 0
    for node in bn.nodes():
        complexity += (bn.F[node]["ri"] - 1) * bn.F[node]["qi"]
    return complexity

def model_complexity(bn, nrows, metric="AIC"):
    if metric == "LL":
        return 0
    elif metric == "AIC":
        return bay_net_size(bn)
    elif metric == "BIC":
        return 0.5 * math.log(nrows) * bay_net_size(bn)

def model_score(data, bn, nodes = None):
    total_score = 0
    if nodes is None:
        nodes = bn.nodes()
    for node in bn.nodes():
        # Create all possible configurations of the parents
        parents = bn.parents(node)
        parent_value_dict = {}
        num_rows = data.shape[0]
        parent_configs = {}
        bn.F[node]["ri"] = len([x for x in np.bincount(data.values[:, data.columns.get_loc(node)]) if x > 0])
        if len(parents) == 0:
            freqs = np.bincount(data.values[:, data.columns.get_loc(node)])
            for count in freqs:
                if count != 0:
                    total_score += count * math.log(count / num_rows)
            bn.F[node]["qi"] = 1
        else:
            str_data = data.values[:, [data.columns.get_loc(p) for p in parents]].astype('str')
            for row in range(num_rows):
                value = '-'.join(str_data[row, 0:len(parents)])
                if value not in parent_value_dict:
                    parent_value_dict[value] = len(parent_value_dict)
                    if not parent_value_dict[value] in parent_configs:
                        parent_configs[parent_value_dict[value]] = []
                parent_configs[parent_value_dict[value]].append(row)
            for parent_config in parent_configs:
                filtered_data = data.values[parent_configs[parent_config], data.columns.get_loc(node)]
                freqs = np.bincount(filtered_data)
                for freq in freqs:
                    if freq > 0:
                        total_score += freq * math.log(freq / len(parent_configs[parent_config]))
                bn.F[node]["qi"] = len(parent_configs[parent_config])
    return total_score

def calc_num_parent_configs(data, parents, configs_cache):
    if len(parents) == 0:
        return 1

    if tuple(parents) not in configs_cache:
        configs_cache[tuple(parents)] = len(data.groupby([x for x in parents]))

    return configs_cache[tuple(parents)]

class hill_climbing:

    def __init__(self, data, nodes):
        self.data = data
        self.nodes = nodes

        self.nrow = len(self.data)
        self.ncol = len(self.nodes)
        self.names = range(self.ncol)



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
            whitelist = []
        for (u,v) in whitelist:
            self.c_dict[u].append(v)
            self.p_dict[v].append(u)
        print("Whitelist", whitelist)

        self.bn = BayesNet(self.c_dict)

        # COMPUTE INITIAL LIKELIHOOD SCORE
    #    value_dict = dict([(n, np.unique(np_data[:,i])) for i,n in enumerate(names)])
        print("Nodes:", list(self.bn.nodes()))

        score = model_score(self.data, self.bn) - model_complexity(self.bn, self.nrow, metric)
        print("Initial Score:", score)

        # CREATE EMPIRICAL DISTRIBUTION OBJECT FOR CACHING
        #ED = EmpiricalDistribution(data,names)

        _iter = 0
        improvement = True

        man = Manager()

        mut_inf_cache = man.dict()
        configs_cache = man.dict()

        while improvement:
            start_t = time.time()
            improvement = False
            max_delta = 0
            max_operation = None

            if debug:
                print('ITERATION: ' , _iter)


            return_queue = Queue()
            p_add = Process(target=self.test_arc_additions, args=(configs_cache, mut_inf_cache, return_queue))
            p_rem = Process(target=self.test_arc_deletions, args=(configs_cache, mut_inf_cache, return_queue))
            p_rev = Process(target=self.test_arc_reversals, args=(configs_cache, mut_inf_cache, return_queue))

            p_add.start()
            p_rem.start()
            p_rev.start()

            p_add.join()
            p_rem.join()
            p_rev.join()

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
                    if debug:
                        print("delta:", max_delta)
                        print('ADDING: ' , str_arc , '\n')
                    self.p_dict[v].append(u)
                    self.bn.add_edge(u,v)
                    self.bn.F[v]["qi"] = max_qi
                elif max_operation == 'Deletion':
                    if debug:
                        print("delta:", max_delta)
                        print('DELETING: ' , str_arc , '\n')
                    self.p_dict[v].remove(u)
                    self.bn.remove_edge(u,v)
                    self.bn.F[v]["qi"] = max_qi
                elif max_operation == 'Reversal':
                    if debug:
                        print("delta:", max_delta)
                        print('REVERSING: ' , str_arc, '\n')
                    self.p_dict[v].remove(u)
                    self.bn.remove_edge(u,v)
                    self.bn.F[v]['qi'] = max_qi[1]
                    self.p_dict[u].append(v)
                    self.bn.add_edge(v,u)
                    self.bn.F[u]['qi'] = max_qi[0]
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

    def test_arc_deletions(self, configs_cache, mut_inf_cache, return_queue):
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
                    if old_cols not in mut_inf_cache:
                        mut_inf_cache[old_cols] = mutual_information(self.data[list(old_cols)])
                    mi_old = mut_inf_cache[old_cols]

                    new_cols = tuple([i for i in old_cols if i != u])  # without 'u' as parent
                    if new_cols not in mut_inf_cache:
                        mut_inf_cache[new_cols] = mutual_information(self.data[list(new_cols)])
                    mi_new = mut_inf_cache[new_cols]

                    ri = self.bn.F[v]['ri']
                    qi = self.bn.F[v]['qi']
                    qi_new = calc_num_parent_configs(self.data, [x for x in self.bn.parents(v) if x != u], configs_cache)
                    delta_score = self.nrow * (mi_new - mi_old) - (ri * (qi_new - qi) - (
                                qi_new - qi))  # Add difference in complexity -> recalculate qi for node v

                    if delta_score - max_delta > 10 ** (-10):
                        max_delta = delta_score
                        max_operation = 'Deletion'
                        max_arc = (u, v)
                        max_qi = qi_new
        return_queue.put((max_arc, max_delta, max_operation, max_qi))

    def test_arc_additions(self, configs_cache, mut_inf_cache, return_queue):
        print("Test Additions")
        ### TEST ARC ADDITIONS ###
        max_delta = 0
        max_operation = None
        max_arc = None
        max_qi = 0
        procs = []
        result_queue = Queue()
        for u in self.bn.nodes():
            p = Process(target=self.test_arcs, args=(configs_cache, mut_inf_cache, u, result_queue))
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
                max_qi = results[3]
        return_queue.put((max_arc, max_delta, max_operation, max_qi))

    def test_arcs(self, configs_cache, mut_inf_cache, u, result_queue):
        max_delta = 0
        max_operation = None
        max_arc = None
        max_qi = 0
        for v in [n for n in self.bn.nodes() if u != n and n not in self.c_dict[u] and not would_cause_cycle(self.c_dict, u, n)]:
            # FOR MMHC ALGORITHM -> Edge Restrictions
            if self.restriction is None or (u, v) in self.restriction:
                # SCORE FOR 'V' -> gaining a parent
                old_cols = (v,) + tuple(self.p_dict[v])  # without 'u' as parent
                if old_cols not in mut_inf_cache:
                    mut_inf_cache[old_cols] = mutual_information(self.data[list(old_cols)])
                mi_old = mut_inf_cache[old_cols]
    
                new_cols = old_cols + (u,)  # with'u' as parent
                if new_cols not in mut_inf_cache:
                    mut_inf_cache[new_cols] = mutual_information(self.data[list(new_cols)])
                mi_new = mut_inf_cache[new_cols]

                ri = self.bn.F[v]['ri']
                qi = self.bn.F[v]['qi']
                qi_new = calc_num_parent_configs(self.data, self.bn.parents(v) + [u], configs_cache)
                delta_score = self.nrow * (mi_new - mi_old) - (ri * (qi_new - qi) - (
                        qi_new - qi))  # Add difference in complexity -> recalculate qi for node v

                if delta_score - max_delta > 10 ** (-10):
                    max_delta = delta_score
                    max_operation = 'Addition'
                    max_arc = (u, v)
                    max_qi = qi_new
        result_queue.put((max_arc, max_delta, max_operation, max_qi))












