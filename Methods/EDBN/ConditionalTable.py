
class ConditionalTable():

    def __init__(self, attr_name):
        self.attr_name = attr_name
        self.parents = []

    def add_parent(self, parent):
        self.parents.append(parent)

    def check_parent_combination(self, parent_combination):
        raise NotImplementedError()

    def get_parent_combinations(self):
        raise NotImplementedError()

    def get_values(self, parent_val):
        raise NotImplementedError()

    def train(self, log):
        raise NotImplementedError()

    def test(self, row):
        raise NotImplementedError()
