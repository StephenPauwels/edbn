import numpy as np
import math
import sklearn.metrics as skm

from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV

from multiprocessing import Manager, Process, Queue

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
                total_complexity += self.numericalComplexity(node)
            elif self.log.isCategoricalAttribute(node):
                total_complexity += self.categoricalComplexity(node)
        return total_complexity

    def categoricalComplexity(self, node):
        """
        Calculate the complexity factor of a particular categorical node
        """
        return self.nodes[node]["qi"]

    def numericalComplexity(self, columns):
        """
        Calculate the complexity factor of a particular numerical node
        """
        complexity = [4,19,67,223,768,2790,10700,43700,187000,842000]
        return complexity[min(len(columns), len(complexity))]

    def model_score(self):
        """
        Calculate the score for the entire model
        """
        total_score = 0

        for node in self.nodes.keys():
            if self.log.isNumericAttribute(node):
                total_score += self.numericalScore(node)
            elif self.log.isCategoricalAttribute(node):
                total_score += self.categoricalScore(node)

        return total_score

    def categoricalScore(self, node):
        """
        Calculate the score for a particular categorical node
        """
        total_score = 0
        num_rows = self.data.shape[0]

        # Create all possible configurations of the parents
        parents = self.p_dict[node]
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
            self.nodes[node]["qi"] = 1
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
            self.nodes[node]["qi"] = self.data.drop_duplicates(list(parents)).shape[0]
        return total_score

    def numericalScore(self, node):
        """
        Calculate the score for a particular numerical node
        """
        print("NumericalScore:", node)
        return self.calc_kde_score([node])

    def categoricalDelta(self, node, parents, mut_inf_cache):
        """
        Score for a particular categorical node given a set of categorical parents
        """
        cols = (node,) + tuple(parents)
        if cols not in mut_inf_cache:
            mut_inf_cache[cols] = mutual_information(self.data[list(cols)])
        mi = mut_inf_cache[cols]
        qi = self.data.drop_duplicates(list(cols)).shape[0]

        return self.nrow * mi - qi, qi


    def numericalDelta(self, node, parents, l_inf_cache):
        """
        Score for a particular numerical node given a set of numerical parents
        """
        cols1 = (node,) + tuple(parents)
        if cols1 not in l_inf_cache:
            l_inf_cache[cols1] = self.calc_kde_score(cols1)

        if len(cols1) == 1:
            return l_inf_cache[cols1]
        else:
            cols2 = tuple(parents)
            if cols2 not in l_inf_cache:
                l_inf_cache[cols2] = self.calc_kde_score(cols2)
            return (l_inf_cache[cols1] - l_inf_cache[cols2]) - self.numericalComplexity(cols1)


    def numericalCategoricalDelta(self, node, parents):
        """
        Score for a particular numerical node given a set of categorical parents
        """
        # TODO
        return 0

    def calc_kde_score(self, cols):
        vals = self.data[list(cols)].values

        # Calculate best bandwith for KDE
        params = {'bandwidth': np.logspace(-2, 5, 20)}
        grid = GridSearchCV(KernelDensity(kernel='gaussian', rtol=1E-6), params, cv=2)
        grid.fit(vals)

        kdens = KernelDensity(kernel='gaussian', bandwidth=grid.best_estimator_.bandwidth, rtol=1E-6).fit(vals)
        return kdens.score(vals)


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
        print("Whitelist", whitelist)
        self.whitelist = whitelist

        print("Nodes:", self.nodes.keys())

        score = self.model_score() - self.model_complexity()
        print("Initial Score:", score)

        _iter = 0
        improvement = True

        man = Manager()

        mut_inf_cache = man.dict()
        l_inf_cache = man.dict()
        configs_cache = man.dict()


        while improvement:
            print("Iteration", _iter)
            improvement = False
            max_delta = 0
            max_operation = None

            return_queue = Queue()
            p_add = Process(target=self.test_arc_additions, args=(configs_cache, mut_inf_cache, return_queue))
            p_rem = Process(target=self.test_arc_deletions, args=(configs_cache, mut_inf_cache, return_queue))

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
                elif max_operation == 'Deletion':
                    self.p_dict[v].remove(u)
                    self.c_dict[u].remove(v)
                    if max_qi is not None:
                        self.nodes[v]["qi"] = max_qi
                print("Model score:", score)
            _iter += 1

        print("SCORE =", score)


    def test_arc_deletions(self, configs_cache, mut_inf_cache, return_queue):
        ### TEST ARC DELETIONS ###
        max_delta = 0
        max_operation = None
        max_arc = None
        max_qi = 0
        for u in self.nodes.keys():
            for v in [n for n in self.c_dict[u] if (u,n) not in self.whitelist]:
                if self.log.isCategoricalAttribute(u) and self.log.isCategoricalAttribute(u):
                    old_score, _ = self.categoricalDelta(v, self.p_dict[v], mut_inf_cache)
                    new_score, new_qi = self.categoricalDelta(v, [i for i in self.p_dict[v] if i != u], mut_inf_cache)
                elif self.log.isNumericAttribute(u) and self.log.isNumericAttribute(v):
                    old_score = self.numericalDelta(v, [par for par in self.p_dict[v] if self.log.isNumericAttribute(par)], mut_inf_cache)
                    new_score = self.numericalDelta(v, [par for par in self.p_dict[v] if par != u and self.log.isNumericAttribute(par)], mut_inf_cache)
                    new_qi = None
                elif self.log.isCategoricalAttribute(u) and self.log.isNumericAttribute(v):
                    old_score = self.numericalCategoricalDelta(v, [par for par in self.p_dict[v] if self.log.isCategoricalAttribute(par)], mut_inf_cache)
                    new_score = self.numericalCategoricalDelta(v, [par for par in self.p_dict[v] if par != u and self.log.isCategoricalAttribute(par)], mut_inf_cache)
                    new_qi = None
                delta_score = new_score - old_score


                if delta_score - max_delta > 10 ** (-10):
                    max_delta = delta_score
                    max_operation = 'Deletion'
                    max_arc = (u, v)
                    max_qi = new_qi
        return_queue.put((max_arc, max_delta, max_operation, max_qi))

    def test_arc_additions(self, configs_cache, mut_inf_cache, return_queue):
        ### TEST ARC ADDITIONS ###
        max_delta = 0
        max_operation = None
        max_arc = None
        max_qi = 0
        procs = []
        result_queue = Queue()
        for u in self.nodes.keys():
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
        for v in [n for n in self.nodes.keys() if
                  u != n and n not in self.c_dict[u] and not would_cause_cycle(self.c_dict, u, n)]:
            # FOR MMHC ALGORITHM -> Edge Restrictions
            if self.restriction is None or (u, v) in self.restriction:
                if self.log.isCategoricalAttribute(u) and self.log.isCategoricalAttribute(u):
                    old_score, _ = self.categoricalDelta(v, self.p_dict[v], mut_inf_cache)
                    new_score, new_qi = self.categoricalDelta(v, self.p_dict[v] + [u], mut_inf_cache)
                elif self.log.isNumericAttribute(u) and self.log.isNumericAttribute(v):
                    old_score = self.numericalDelta(v, [par for par in self.p_dict[v] if self.log.isNumericAttribute(par)], mut_inf_cache)
                    new_score = self.numericalDelta(v, [par for par in self.p_dict[v] + [u] if self.log.isNumericAttribute(par)], mut_inf_cache)
                    new_qi = None
                elif self.log.isCategoricalAttribute(u) and self.log.isNumericAttribute(v):
                    old_score, _ = self.numericalCategoricalDelta(v, [par for par in self.p_dict[v] if self.log.isCategoricalAttribute(par)], mut_inf_cache)
                    new_score, new_qi = self.numericalCategoricalDelta(v, [par for par in self.p_dict[v] + [u] if self.log.isCategoricalAttribute(par)], mut_inf_cache)
                    new_qi = None
                delta_score = new_score - old_score


                if delta_score - max_delta > 10 ** (-10):
                    max_delta = delta_score
                    max_operation = 'Addition'
                    max_arc = (u, v)
                    max_qi = new_qi
        result_queue.put((max_arc, max_delta, max_operation, max_qi))



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


if __name__ == "__main__":
    from LogFile import *

    # train_data = LogFile("../Data/bpic15_5_train.csv", ",", 0, 500000, time_attr="Complete_Timestamp", trace_attr="Case_ID",activity_attr="Activity")
    # train_data.create_k_context()
    # attributes = ['Activity', 'Resource', 'Weekday']
    #
    # restrictions = []
    # for attr1 in attributes:
    #     for attr2 in attributes:
    #         if attr1 != attr2:
    #             restrictions.append((attr2, attr1))
    #         for i in range(1):
    #             restrictions.append((attr2 + "_Prev%i" % (i), attr1))
    #
    # learner = Structure_learner(train_data, ['Activity', 'Resource', 'Weekday', 'Activity_Prev0', 'Resource_Prev0', 'Weekday_Prev0'])
    # learner.learn(restrictions=restrictions)

    train_data = LogFile("../Data/creditcard.csv", ",", 0, 1000, time_attr=None, trace_attr=None, convert=False)

    train_data.data = train_data.data[train_data.data.Class == "0"] # Only keep non-anomalies
    train_data.remove_attributes(["Time", "Class", "Amount"])

    learner = Structure_learner(train_data, train_data.data.columns)
    learner.learn()

    # -506813.5041904236
    # -401821.717691689

    # SCORE = -312447.16749336576
    # SCORE2 = -311246.1674933654

    # SCORE = -312809.1674933657
    # SCORE2 = -311246.1674933652


    # SCORE = -12395.453915026774
    # SCORE2 = -30997.018277085794