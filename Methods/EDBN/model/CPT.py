from Methods.EDBN.model.ConditionalTable import ConditionalTable

class CPT(ConditionalTable):

    def __init__(self, attr_name):
        super().__init__(attr_name)

        self.cpt = {}
        self.cpt_probs = dict()
        self.new_relations = None
        self.parent_count = dict()

    def add_parent(self, parent):
        self.parents.append(parent)

    def remove_parent(self, parent):
        self.parents.remove(parent)

    def check_parent_combination(self, parent_combination):
        return parent_combination in self.cpt

    def get_parent_combinations(self):
        return self.cpt.keys()

    def get_values(self, parent_val):
        output = {}
        if len(parent_val) == 1:
            parent_val = parent_val[0]
        for val, count in self.cpt[parent_val].items():
            output[val] = count / self.parent_count[parent_val]
        return output
        # return self.cpt[parent_val]


    def train(self, log):
        self.cpt = {}
        self.cpt_probs = dict()
        self.new_relations = None
        self.parent_count = dict()

        self.learn_table(log)
        self.set_new_relation(log)

    def learn_table2(self, log):
        if len(self.parents) == 0:
            return

        parents = [p.attr_name for p in self.parents]
        grouped = log.groupby(parents, observed=True)[self.attr_name]
        val_counts = grouped.value_counts()
        div = grouped.count().to_dict()
        total_rows = len(log.index)
        for t in val_counts.items():
            parent = t[0][:-1]
            if parent not in self.cpt:
                self.cpt[parent] = dict()
            if len(parent) == 1:
                self.cpt[parent][t[0][-1]] = t[1] / div[parent[0]]
                self.cpt_probs[parent] = div[parent[0]] / total_rows
            else:
                self.cpt[parent][t[0][-1]] = t[1] / div[parent]
                self.cpt_probs[parent] = div[parent] / total_rows

    def learn_table(self, log):
        if len(self.parents) == 0:
            return

        parents = [p.attr_name for p in self.parents]
        grouped = log.groupby(parents, observed=True)[self.attr_name]
        val_counts = grouped.value_counts()
        self.parent_count = grouped.count().to_dict()
        total_rows = len(log.index)
        for t in val_counts.items():
            parent = t[0][:-1]
            if len(parent) == 1:
                parent = parent[0]
            if parent not in self.cpt:
                self.cpt[parent] = dict()

            self.cpt[parent][t[0][-1]] = t[1]
            self.cpt_probs[parent] = 0

    def set_new_relation(self, log):
        attrs = set()
        if len(self.parents) == 0:
            self.new_relations = 1
            return
        attrs = {p.attr_name for p in self.parents}
        grouped = log.groupby([a for a in attrs]).size().reset_index(name='counts')
        self.new_relations = len(grouped) / log.shape[0]

    def update(self, row):
        parent_val = tuple([getattr(row, attr.attr_name) for attr in self.parents])
        if len(parent_val) == 1:
            parent_val = parent_val[0]
        value = getattr(row, self.attr_name)
        if parent_val in self.parent_count:
            self.parent_count[parent_val] += 1
        else:
            self.parent_count[parent_val] = 1
            self.cpt[parent_val] = dict()

        if value in self.cpt[parent_val]:
            self.cpt[parent_val][value] += 1
        else:
            self.cpt[parent_val][value] = 0

    def test(self, row):
        if len(self.parents) > 0:
            parent_vals = []
            for p in self.parents:
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
            return (1 - self.new_relations) * (self.cpt[parent_vals][val] / self.parent_count[parent_vals])
        return 1

