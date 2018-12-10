import numpy as np
import matplotlib.pyplot as plt
from LogFile import LogFile
import pandas as pd
import math
import statsmodels.api as sm
import time
import kde
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV

from pyfigtree import figtree

def event_score(value, kernel):
    return math.log(kernel(value))


def variable_score(variable, parents, data):
    score = 0
    if len(parents) == 0:
        #print(data)
        column = data[variable]
        #print(column)
        #kernel = kde.gaussian_kde(column.values)
        #
        #x = np.linspace(min(column.values), max(column.values), 1000)
        #print(kernel.covariance_factor())
        #plt.plot(x, np.log(kernel(x)))
        #plt.show()
        #sample = kernel.resample(5000)
        #kernel = kde.gaussian_kde(sample)
        #plt.plot(x, kernel(x))
        #plt.show()
        #start = time.time()
        #print(kernel.logpdf(column.values).sum())
        #print("scipy: ", time.time() - start)

        #grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1,1.0,10)}, cv=10)
        #grid.fit(column.values[:, None])
        #print(grid.best_params_)

        vals = column.values[:, np.newaxis]

        #x = np.linspace(min(column.values), max(column.values), 1000)
        #kdens = KernelDensity(kernel='gaussian', bandwidth=1, rtol=0).fit(vals)
        #plt.plot(x, kdens.score_samples(x[:, np.newaxis]))
        #plt.show()

        start = time.time()
        kdens = KernelDensity(kernel='gaussian', bandwidth=0.2, rtol=1E-2).fit(vals)
        plt.plot(sorted(vals, reverse=True), kdens.score_samples(sorted(vals, reverse=True)))
        plt.show()
        print(kdens.score(vals))
        print("sklearn: ", time.time() - start)



        #array = np.unique(data[variable].values)
        #plt.scatter(array, [0] * len(array))
        #plt.plot(np.linspace(min(array), max(array), 1000), kernel(np.linspace(min(array), max(array), 1000)) )
        #plt.show()

        #start = time.time()
        #print(column.apply(event_score, args=(kernel,)).sum())
        #print("apply: ", time.time() - start)

        #start = time.time()
        #density = sm.nonparametric.KDEMultivariate(data=[column], var_type='c')
        #print(len(column.values), len(np.unique(column.values)))
        #print(np.log(density.pdf(column.values)).sum())
        #print("statsmodels: ", time.time() -  start)
    else:
        cols = parents + [variable]
        d = data[cols]
        #print(d)
        #print(d.values)
        samp = KernelDensity(kernel='gaussian', bandwidth=0.2, rtol=1E-8).fit(d.values).sample(5000)
        score1 = KernelDensity(kernel='gaussian', bandwidth=0.2, rtol=1E-8).fit(samp).score(d.values)
        samp = KernelDensity(kernel='gaussian', bandwidth=0.2, rtol=1E-8).fit(data[parents].values).sample(5000)
        score2 = KernelDensity(kernel='gaussian', bandwidth=0.2, rtol=1E-8).fit(samp).score(data[parents].values)
        print(variable, parents, score1, score2, score1 - score2)
        return score1 - score2
        #print(KernelDensity(bandwidth=0.2).fit([np.linspace(-5,5, 100)]).score_samples([np.linspace(-5,5, 100)]))
        #plt.plot(np.linspace(-5, 5, 100), KernelDensity(bandwidth=0.2).fit([np.linspace(-5,5, 100)]).score_samples([np.linspace(-5,5, 100)]))
        #plt.show()
    return score



def variable_delta(variable, parents, data):
    pass



def test(data, cols, bandwidth):
    if len(cols) == 0:
        return 0


    samples = data[cols].values

    weights = np.ones(len(samples)) / len(samples)# / np.sqrt(np.pi)# / bandwidth

    target_densities = figtree(samples, samples, weights, bandwidth=bandwidth)
    s = np.sum(np.log(target_densities))

    #print("Score:", s)
    return s



def test2(data, col):
    vals = data[col].values
    kdens = KernelDensity(kernel='gaussian', bandwidth=0.5, rtol=1E-2).fit(vals)
    s = kdens.score(vals)
    print("Score:", s)
    return s

def would_cause_cycle(child, parent, parents):
    import copy
    parents_new = copy.deepcopy(parents)
    parents_new[child].append(parent)
    visited = []
    to_visit = parents_new[child]
    while len(to_visit) > 0:
        for node in to_visit:
            to_visit.extend(parents_new[node])
            if node in visited:
                return True
            else:
                visited.append(node)
            to_visit.remove(node)
    return False

def calc_bandwidth(x):
    from scipy.stats import scoreatpercentile as sap
    normalize = 1.349
    IQR = (sap(x, 75) - sap(x,25)) / normalize
    sigma = np.minimum(np.std(x, axis=0, ddof=1), IQR)
    return max(0.01, 0.9 * sigma * len(x) ** (-0.2))

if __name__ == "__main__":
    #log = pd.read_csv("../Data/creditcard.csv", nrows=20000, dtype='float64')
    #columns = ["V1", "V2", "V3", "V4", "V5", "Time", "Amount"]
    #parents = [[], [], [], [], [], [], []]

    log = pd.read_csv("../Data/boston_train.csv", nrows=20000, dtype='float64')
    log = log.drop(columns=["ID"])
    columns = log.columns


    # Calculate log-likelihood score for entire dataset without dependencies
    parents = {}
    bandwidth = {}
    score = 0
    for col in columns:
        bandwidth[col] = calc_bandwidth(log[col])
        score += test(log, [col], bandwidth[col])
        parents[col] = []

    print("Model Score:", score)
    print(bandwidth)



    max_delta = -1
    while max_delta != 0:
        #print(parents)
        max_delta = 0
        max_operation = ""
        max_edge = None
        for col1 in columns:
            bw = bandwidth[col1]

            # Test addition of new parent
            for col2 in columns:
                if col2 != col1 and col2 not in parents[col1] and not would_cause_cycle(col1, col2, parents):
                    new_parents = parents[col1] + [col2]
                    old_parents = parents[col1]

                    par_size = 0
                    for p in parents:
                        par_size += len(parents[p])
                    delta_s = test(log, new_parents + [col1], bw) - test(log, new_parents, bw)
                    delta_s -= test(log, old_parents + [col1], bw) - test(log, old_parents, bw)
                #    delta_s *= (0.1 * (par_size + 1)) / (0.1 * par_size)

                    #print(delta_s)
                    if delta_s > max_delta:
                        max_delta = delta_s
                        max_edge = (col2, col1)
                        max_operation = "Add"

            for p in parents[col1]:
                old_parents = parents[col1]
                new_parents = parents[col1][:]
                new_parents.remove(p)

                #print(new_parents, old_parents)
                delta_s = test(log, new_parents + [col1], bw) - test(log, new_parents, bw)
                delta_s -= test(log, old_parents + [col1], bw) - test(log, old_parents, bw)

                if delta_s > max_delta:
                    max_delta = delta_s
                    max_edge = (col2, col1)
                    max_operation = "Remove"

        print("delta=", max_delta, " max=", max_edge, " operation=", max_operation)
        if max_delta != 0:
            if max_operation == "Add":
                parents[max_edge[1]].append(max_edge[0])
            elif max_operation == "Remove":
                parents[max_edge[1]].remove(max_edge[0])

    print(parents)



    """
    for nrow in [20000]:

        variable_score("V1", [], log)

        results = []
        for c1 in ["V1", "V2"]:
            for c2 in ["V3", "V4", "V5", "V6"]:
                results.append((c1, c2, variable_score(c1, [c2], log)))
        for res in sorted(results, key=lambda l : l[2]):
            print(res)
    """




"""
# Add arc from event_Prev0 to event
print("Creating Kernels")
kernel_comb_old = stats.gaussian_kde(log.contextdata["event"])
kernel_parent_old = None
kernel_comb_new = stats.gaussian_kde([log.contextdata["event"], log.contextdata["event_Prev0"]])
kernel_parent_new = stats.gaussian_kde(log.contextdata["completeTime"])

print("Calculating scores")
S_comb_old = np.sum(np.log10(kernel_comb_old(log.contextdata["event"])))
S_comb_new = np.sum(np.log10(kernel_comb_new([log.contextdata["event"], log.contextdata["event_Prev0"]])))
S_parent_new = np.sum(np.log10(kernel_parent_new(log.contextdata["event_Prev0"])))
"""


