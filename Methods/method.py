import copy
import time

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
        timings = []
        for predict_time in range(len(data.get_batch_ids())):
            print("%i / %i" % (predict_time, len(data.get_batch_ids())))
            predict_batch = data.get_test_batch(predict_time)
            # Test current batch
            results.extend(self.test(model, predict_batch))

            start_time = time.time()
            # Prepare new train-data
            if window == 0: # Window == 0 -> Use all available historical events
                train_data = train_data.extend_data(predict_batch)
            elif window >= 1:
                train_data = predict_batch
                for w in range(1,window):
                    add_time = predict_time - w
                    if add_time < 0:
                        train_data = train_data.extend_data(data.train)
                        break
                    else:
                        train_data = train_data.extend_data(data.get_test_batch(add_time))


            # Update
            if reset:
                model = self.train(train_data)
            else:
                model = self.update(model, train_data)

            timings.append(time.time() - start_time)
        return results, timings

    def test_and_update_drift(self, model, data, drifts, reset):
        try:
            model = copy.deepcopy(model)
        except:
            import tensorflow as tf
            model.save("tmp_model")
            model = tf.keras.models.load_model("tmp_model")

        train_data = data.train

        results = []
        timings = []
        for predict_time in range(len(data.get_batch_ids())):
            print("%i / %i" % (predict_time, len(data.get_batch_ids())))
            predict_batch = data.get_test_batch(predict_time)
            # Test current batch
            results.extend(self.test(model, predict_batch))

            start_time = time.time()
            if reset:
                if predict_time in drifts:  # New drift detected
                    print("RESET - Drift Detected")
                    train_data = predict_batch
                else:
                    print("RESET - No drift")
                    train_data = train_data.extend_data(predict_batch)
                model = self.train(train_data)
            else:
                train_data = predict_batch
                if predict_time in drifts:
                    print("UPDATE - Drift Detected")
                    model = self.train(train_data)
                else:
                    print("UPDATE - No drift")
                    model = self.update(model, train_data)
            timings.append(time.time() - start_time)

        return results, timings

    def k_fold_validation(self, data):
        import pickle

        for i in range(1,len(data.folds)):
            print(i, "/", len(data.folds))
            data.get_fold(i)
            model = self.train(data.train, self.def_params)
            # with open("k_result_%i" % i, "rb") as finn:
            #     new_results = pickle.load(finn)
            # results.extend(new_results)
            # results.extend()
            results = self.test(model, data.test)
            with open("k_result_%i" % i, "wb") as fout:
                pickle.dump(results, fout)

        results = []
        for i in range(len(data.folds)):
            with open("k_result_%i" % i, "rb") as finn:
                new_results = pickle.load(finn)
            results.extend(new_results)
        return results

