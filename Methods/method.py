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

    def update_index(self, model, train_data, params=None):
        return self.update_func(model, train_data)

    def test_index(self, model, test_data):
        return self.test_func(model, test_data)

    def test_and_update_indices(self, model, data, evWindow, prevTestInd, currentTestInd, reset):
        '''Windows have been set to size of 0 months, 1 month,  and 5 months
           for how much historical data to use in training/retraining.
        '''
        try:
            model = copy.deepcopy(model)
        except:
            import tensorflow as tf
            model.save("tmp_model")
            model = tf.keras.models.load_model('tmp_model')

        # train_data = data.train

        # results = []
        timings = []
        # for predict_time in range(len(indices)):
        # print("%i / %i" % (evWindow, len(data.get_batch_ids())))
        predict_batch = data.get_test_batchi(prevTestInd, currentTestInd)
        # Test current batch
        print("Predict batch size")
        print(predict_batch.contextdata.shape)
        results = self.test(model, predict_batch)

        start_time = time.time()
        # Prepare new train-data
        # if evWindow == 0: # Window == 0 -> Use all available historical events
        #     train_data = train_data.extend_data(predict_batch)
        # elif evWindow >= 1:
        train_data = data.get_test_batchi(evWindow[1], evWindow[0])
        #This needs to be re-worked to accommodate new indexing type (only else statement)

            # add_time = evWindow[1] - evWindow
            # if add_time < 0:
            #     train_data = train_data.extend_data(data.train)
            #     break
            # else:
        # train_data = train_data.extend_data(data.get_test_batchi(add_time, predict_time[1]))


        # Update
        if reset:
            model = self.train(train_data)
        else:
            model = self.update(model, train_data)

        timings = time.time() - start_time
        return results, timings, model

    def test_and_update(self, model, data, window, reset):
        '''Windows have been set to size of 0 months, 1 month,  and 5 months
           for how much historical data to use in training/retraining.
        '''
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
        '''Drifts have been pre-determined from dataset and retraining occurs at
           these known drift points.
        '''
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

            ### TODO: Change Below so that it can accept new data as input from incoming data
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

    def test_and_update_drift_index(self, model, data, indices, reset):
        '''Drifts have been pre-determined from dataset and retraining occurs at
           these known drift points.
        '''
        try:
            model = copy.deepcopy(model)
        except:
            import tensorflow as tf
            model.save("tmp_model")
            model = tf.keras.models.load_model("tmp_model")

        train_data = data.train

        results = []
        timings = []
        #indices should be a list of two pair tuple index ranges
        #[(idx1, idx2), (i2dx1, i2dx2)]
        for predict_time in range(len(indices)):
            print("%i / %i" % (predict_time, len(data.get_batch_ids())))

            ### TODO: Change Below so that it can accept new data as input from incoming data
            predict_batch = data.get_test_batchi(predict_time[0], predict_time[1])
            # Test current batch
            results.extend(self.test(model, predict_batch))

            start_time = time.time()
            train_data = predict_batch
            #For drifts, retrain
            # print("UPDATE - Drift Detected")
            # model = self.train(train_data)
            #Otherwise no drift and Update
            print("UPDATE - No drift")
            model = self.update(model, train_data)
            timings.append(time.time() - start_time)

        return results, timings

    # NEED TO EDIT THIS FOR THE NEW UPDATING OF THE WINDOWS.......
    # def test_and_update_drift_adwin(self, model, data, reset):
    #     '''Drift is automatically detected and corrected for once it is noticed.
    #        this occurs using an ADWIN method.
    #     '''
    #     try:
    #         model = copy.deepcopy(model)
    #     except:
    #         import tensorflow as tf
    #         model.save("tmp_model")
    #         model = tf.keras.models.load_model("tmp_model")
    #
    #     train_data = data.train
    #
    #     results = []
    #     timings = []
    #     for predict_time in range(len(data.get_batch_ids())):
    #         print("%i / %i" % (predict_time, len(data.get_batch_ids())))
    #         predict_batch = data.get_test_batch(predict_time)
    #         # Test current batch
    #         results.extend(self.test(model, predict_batch))
    #
    #         start_time = time.time()
    #         if reset:
    #             if predict_time in drifts:  # New drift detected
    #                 print("RESET - Drift Detected")
    #                 train_data = predict_batch
    #             else:
    #                 print("RESET - No drift")
    #                 train_data = train_data.extend_data(predict_batch)
    #             model = self.train(train_data)
    #         else:
    #             train_data = predict_batch
    #             if predict_time in drifts:
    #                 print("UPDATE - Drift Detected")
    #                 model = self.train(train_data)
    #             else:
    #                 print("UPDATE - No drift")
    #                 model = self.update(model, train_data)
    #         timings.append(time.time() - start_time)
    #
    #     return results, timings

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

