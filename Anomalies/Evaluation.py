"""
    File containing evaluation of the model on synthetic data

    Author: Stephen Pauwels
"""

import numpy as np

import Methods.EDBN.Anomalies as edbn
import Utils.DataDurationGenerator as duration_generator
import Utils.DataGenerator as generator
import Utils.PlotResults as plt
from Utils.LogFile import LogFile

RUNS = 10

def duration_test():
    path = "../Data/Experiments_Duration/"
    train_rates = [0,5,10,25]
    test_rates = [1,5,10,25,50,100,250,500]
    anoms_rates = []
    for train_rate in train_rates:
        for test_rate in test_rates:
            anoms_rates.append((train_rate, test_rate))

    for i in range(len(anoms_rates)):
        print(anoms_rates[i])
        scores = []
        for run in range(RUNS):
            print("Run %i" % run)
            train_file = path + "%i_train_%i.csv" % (i, anoms_rates[i][0])
            test_file = path + "%i_test_%i.csv" % (i, anoms_rates[i][1])
            duration_generator.generate(10000, 10000, anoms_rates[i][0], anoms_rates[i][1], train_file, test_file)

            train_data = LogFile(train_file, ",", 0, 1000000, "date", "trace")
            train_data.remove_attributes(["Anomaly"])
            test_data = LogFile(test_file, ",", 0, 1000000, "date", "trace", values=train_data.values)

            train_data.keep_attributes(["event", "date", "trace", "process", "resource", "random"])

            train_data.create_k_context()
            train_data.add_duration_to_k_context()
            bins = train_data.discretize("duration_0")
            test_data.create_k_context()
            test_data.add_duration_to_k_context()
            test_data.discretize("duration_0", bins)

            model = edbn.train(train_data)
            edbn.test(test_data, path + "Output_%i_%i.csv" % anoms_rates[i], model, "anomaly", "0")

            output_file = path + "Output_%i_%i.csv" % anoms_rates[i]
            output_roc = path + "roc_%i_%i.png" % anoms_rates[i]
            output_prec = path + "prec_recall_%i_%i.png" % anoms_rates[i]

            score = plt.get_roc_auc(output_file)
            scores.append(plt.get_roc_auc(output_file))
            print("Score = %f" % score)

        with open(path + "results.txt", "a") as fout:
            fout.write("Testing:\ntrain rate: %i\ntest rate: %i\n" % (anoms_rates[i][0], anoms_rates[i][1]))
            fout.write("Result: " + str(scores) + "\n")
            fout.write("Mean: %f Median: %f\n" % (np.mean(scores), np.median(scores)))
            fout.write("Variance: %f\n\n" % np.var(scores))

def duration_test_discretize():
    path = "../Data/Experiments_Discretize/"
    train_rates = [0,5,10,25]
    test_rates = [1,5,10,25,50,100,250,500]
    anoms_rates = []
    for train_rate in train_rates:
        for test_rate in test_rates:
            anoms_rates.append((train_rate, test_rate))

    for i in range(len(anoms_rates)):
        print(anoms_rates[i])
        scores = []
        for run in range(RUNS):
            print("Run %i" % run)
            train_file = path + "%i_train_%i.csv" % (i, anoms_rates[i][0])
            test_file = path + "%i_test_%i.csv" % (i, anoms_rates[i][1])
            duration_generator.generate(10000, 10000, anoms_rates[i][0], anoms_rates[i][1], train_file, test_file)

            train_data = LogFile(train_file, ",", 0, 1000000, "date", "trace", convert=False)
            train_data.remove_attributes(["Anomaly"])

            train_data.keep_attributes(["event", "date", "trace", "process", "resource", "random"])
            train_data.convert2int()

            train_data.create_k_context()
            train_data.add_duration_to_k_context()
            bins = train_data.discretize("duration_0", bins=10)

            test_data = LogFile(test_file, ",", 0, 1000000, "date", "trace", values=train_data.values, convert=False)
            test_data.keep_attributes(["event", "date", "trace", "process", "resource", "random", "anomaly"])
            test_data.convert2int()

            test_data.create_k_context()
            test_data.add_duration_to_k_context()
            test_data.discretize("duration_0", bins)

            model = edbn.train(train_data)
            edbn.test(test_data, path + "Output_%i_%i.csv" % anoms_rates[i], model, "anomaly", "0")

            output_file = path + "Output_%i_%i.csv" % anoms_rates[i]
            output_roc = path + "roc_%i_%i.png" % anoms_rates[i]
            output_prec = path + "prec_recall_%i_%i.png" % anoms_rates[i]

            score = plt.get_roc_auc(output_file)
            scores.append(plt.get_roc_auc(output_file))
            print("Score = %f" % score)

        with open(path + "results.txt", "a") as fout:
            fout.write("Testing:\ntrain rate: %i\ntest rate: %i\n" % (anoms_rates[i][0], anoms_rates[i][1]))
            fout.write("Result: " + str(scores) + "\n")
            fout.write("Mean: %f Median: %f\n" % (np.mean(scores), np.median(scores)))
            fout.write("Variance: %f\n\n" % np.var(scores))

def categorical_test():
    path = "../Data/Experiments/"
    train_rates = [0,5,10,25]
    test_rates = [1,5,10,25,50,100,250,500]
    anoms_rates = []
    for train_rate in train_rates:
        for test_rate in test_rates:
            anoms_rates.append((train_rate, test_rate))

    for i in range(len(anoms_rates)) :
        print(anoms_rates[i])
        scores = []
        for run in range(RUNS):
            print("Run %i" % run)
            train_file = path + "%i_train_%i.csv" %(i, anoms_rates[i][0])
            test_file = path + "%i_test_%i.csv" %(i, anoms_rates[i][1])
            generator.create_shipment_data(10000, 10000, anoms_rates[i][0], anoms_rates[i][1], train_file, test_file)

            train_data = LogFile(train_file, ",", 0, 1000000, None, "Case")
            train_data.remove_attributes(["Anomaly"])
            test_data = LogFile(test_file, ",", 0, 1000000, None, "Case",values=train_data.values)

            model = edbn.train(train_data)
            edbn.test(test_data, path + "Output_%i_%i.csv" % anoms_rates[i], model, "Anomaly", "0")

            output_file = path + "Output_%i_%i.csv" % anoms_rates[i]
            output_roc = path + "roc_%i_%i.png" % anoms_rates[i]
            output_prec = path + "prec_recall_%i_%i.png" % anoms_rates[i]

            score = plt.get_roc_auc(output_file)
            scores.append(plt.get_roc_auc(output_file))
            print("Score = %f" % score)


        with open(path + "results.txt", "a") as fout:
            fout.write("Testing:\ntrain rate: %i\ntest rate: %i\n" % (anoms_rates[i][0], anoms_rates[i][1]))
            fout.write("Result: " + str(scores) + "\n")
            fout.write("Mean: %f Median: %f\n" % (np.mean(scores), np.median(scores)))
            fout.write("Variance: %f\n\n" % np.var(scores))

            #plt.plot_single_roc_curve(output_file, output_roc)
            #plt.plot_single_prec_recall_curve(output_file, None, output_prec)


if __name__ == "__main__":
    #categorical_test()
    #duration_test()
    duration_test_discretize()

