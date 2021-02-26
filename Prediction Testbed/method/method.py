import copy

class Method:

    def __init__(self, name, train, test, update=None, def_params=None):
        self.name = name
        self.train_func = train
        self.test_func = test
        self.update_func = update
        self.def_params = def_params

    def __str__(self):
        return "Method: %s" % self.name

    def train(self, train_data, params=None):
        if not params:
            params = self.def_params
        if params:
            return self.train_func(train_data, **params)
        else:
            return self.train_func(train_data)

    def update(self, model, train_data, params=None):
        return self.update_func(model, train_data)

    def test(self, model, test_data):
        return self.test_func(model, test_data)

    def test_and_update(self, model, data, window, reset):
        try:
            model = copy.deepcopy(model)
        except:
            import tensorflow as tf
            model.save("tmp_model")
            model = tf.keras.models.load_model('tmp_model')

        train_data = data.train

        results = []
        for predict_time in range(len(data.get_batch_ids())):
            print("%i / %i" % (predict_time, len(data.get_batch_ids())))
            predict_batch = data.get_test_batch(predict_time)
            # Test current batch
            results.extend(self.test(model, predict_batch))

            # Prepare new train-data
            if window == 0: # Window == 0 -> Use all available historical events
                train_data = data.train.extend_data(predict_batch)
            elif window == 1: # Window == 1 -> Use current tested batch for training
                train_data = predict_batch

            # Update
            if reset:
                model = self.train(train_data)
            else:
                model = self.update(model, train_data)
        return results


    def k_fold_validation(self, data, k):
        data.create_folds(k)

        results = []
        for i in range(k):
            data.get_fold(i)
            self.train(data, self.def_params)
            results.extend(self.test(data))
        return results

