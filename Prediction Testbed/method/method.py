
class Method:

    def __init__(self, name, train, test):
        self.name = name
        self.train_func = train
        self.test_func = test

        self.model = None

    def __str__(self):
        return "Method: %s" % self.name

    def train(self, data, params):
        self.model = self.train_func(data.train, **params)

    def test(self, data, metric):
        return metric.calculate(self.test_func(data.test, self.model))