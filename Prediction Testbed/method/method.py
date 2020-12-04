
class Method:

    def __init__(self, name, train, test, def_params=None):
        self.name = name
        self.train_func = train
        self.test_func = test
        self.def_params = def_params

        self.model = None

    def __str__(self):
        return "Method: %s" % self.name

    def train(self, data, params=None):
        if not params:
            params = self.def_params
        if params:
            self.model = self.train_func(data.train, **params)
        else:
            self.model = self.train_func(data.train)

    def test(self, data, metric):
        return metric.calculate(self.test_func(data.test, self.model))

