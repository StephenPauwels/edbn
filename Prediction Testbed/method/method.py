import copy

class Method:

    def __init__(self, name, train, test, test_and_update=None, test_and_update_retain=None, def_params=None):
        self.name = name
        self.train_func = train
        self.test_func = test
        self.test_and_update_func = test_and_update
        self.test_and_update_retain_func = test_and_update_retain
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

    def test(self, data):
        return [self.test_func(data.test[t]["data"], self.model) for t in data.test]

    def test_and_update(self, data, retain=False):
        if self.model is None:
            first_key = sorted(data.test.keys())[0]
            if self.name == "Di Mauro":
                self.model = self.train_func(data.test[first_key]["data"], **{"early_stop": 4, "params": {"n_modules": 2}})
            else:
                self.model = self.train_func(data.test[first_key]["data"], **self.def_params)
        try:
            model = copy.deepcopy(self.model)
        except:
            import tensorflow as tf
            self.model.save("tmp_model")
            model = tf.keras.models.load_model('tmp_model')

        if self.test_and_update_func:
            if retain and self.test_and_update_retain_func:
                return [self.test_and_update_retain_func(data.test, model, data.train)]
                # results = []
                # train_batch = [data.train]
                # i = 0
                # for t in data.test:
                #     print(i, "/", len(data.test))
                #     results.extend(self.test_and_update_retain_func(data.test[t]["data"], model, train_batch))
                #     train_batch.append(data.test[t]["data"])
                #     i += 1
                # return results
            else:
                return [self.test_and_update_func(data.test, model)]
        return None

