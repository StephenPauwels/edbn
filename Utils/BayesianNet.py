# File for constructing a Bayesian Network
import copy
import sys

import pandas as pd
import sklearn.metrics as skm
from scipy.stats import chi2_contingency

import pyBN.learning.structure as pybn


## Operations on graphs are performed using functions -> Possible refactoring towards maps made easier
# from -> cols
# to -> rows
def create_empty_graph(size):
    graph = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(0)
        graph.append(row)
    return graph


def set_value(graph, col, row, value):
    graph[col][row] = value


class BayesianNetwork:

    def __init__(self, data):
        self.data = data

    #####
    ### Use library pyBN
    def grow_shrink_pybn(self, restrictions=None):
        return pybn.gs(self.data.values, debug=True)

    def hill_climbing_pybn(self, nodes, whitelist=None, restrictions=None, metric = "AIC"):
        # for now: convert data to nested numpy array
        #pybn.gs(self.data.values, alpha=0.01, debug = True)
        print("Learning Bayesian Network")
        hc_algo = pybn.hill_climbing(self.data, nodes)
        return hc_algo.hc(restriction=restrictions, whitelist=whitelist, metric=metric, debug=True)

    #####
    ### Functions needed when building a network using Hill Climbing and scoring
    def hill_climbing(self, whitelist = None, blacklist = None, maxiter = sys.maxsize):
        nodes = list(self.data.columns)
        iteration = 1
        # nodes to be updates (all of them in first iteration)
        updates = list(range(len(nodes)))

        # convert blacklist to adjacency matrix
        black_matrix = create_empty_graph(len(nodes))
        if blacklist:
            for entry in blacklist:
                set_value(black_matrix, nodes.index(entry[1]), nodes.index(entry[0]), 1)

        # convert whitelist to adjacency matrix
        white_matrix = create_empty_graph(len(nodes))
        if whitelist:
            for entry in whitelist:
                set_value(white_matrix, nodes.index(entry[1]), nodes.index(entry[0]), 1)

        # create start graph in adjacency matrix form
        start = copy.deepcopy(white_matrix)

        while(iteration < 10):

            iteration += 1



    #####
    ### Functions needed when building a network using constraints (dependency)
    def independence_test(self, attr1, attr2, conditioned):
        return skm.mutual_info_score(self.data[attr1], self.data[attr2])

    def chisq_test(self, col1, col2):
        groupsizes = self.data.pivot_table(index=[col1, col2], aggfunc=len)
        ctsum = groupsizes.unstack(col1)
        ctsum = ctsum.fillna(0)
        return chi2_contingency(ctsum, correction=False)

    def markov_blanket(self, attr):
        pass

if __name__ == "__main__":
    print("Testing")

    data = pd.read_csv("/Users/Stephen/git/interactive-log-mining/CorrectCases_ints", delimiter=",", nrows=10000, header=None, dtype=int, skiprows=0)
    data.fillna("", inplace=True)
    data.columns = ['pID', 'pTime', 'pIP', 'pType', 'pActivity', 'pOwID', 'pOwName', 'pOwRole', 'pExID', 'pExName',
                    'pExRole', 'pCase', 'ID', 'Time', 'IP', 'type', 'activity', 'owID', 'owName', 'owRole', 'exID',
                    'exName', 'exRole', 'Case']

    for d in ['pTime', 'pCase', 'Time', 'Case', 'pID', 'ID']:
        del data[d]

    blacklist = [['pIP', 'pType']]

    bn = BayesianNetwork(data.iloc[range(10000),:])
    #bn.grow_shrink_pybn()
    bn.hill_climbing_pybn()


    print("Testing Done")