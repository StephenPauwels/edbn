import eDBN.Execute as edbn
import Utils.DataGenerator as generator
import Utils.PlotResults as plt
import Utils.Utils as utils
from Utils.LogFile import LogFile

import numpy as np

if __name__ == "__main__":
    path = "../Data/Experiments/"
    train_rates = [0]#,5,10,25]
    test_rates = [1]#,5,10,25,50,100,250,500]
    anoms_rates = []
    for train_rate in train_rates:
        for test_rate in test_rates:
            anoms_rates.append((train_rate, test_rate))

    for i in range(len(anoms_rates)) :
        print(anoms_rates[i])
        scores = []
        for run in range(10):
            print("Run %i" % run)
            train_file = path + "%i_train_%i.csv" %(i, anoms_rates[i][0])
            test_file = path + "%i_test_%i.csv" %(i, anoms_rates[i][1])
            generator.create_shipment_data(10000, 10000, anoms_rates[i][0], anoms_rates[i][1], train_file, test_file)

            #dict_dict = []
            #utils.convert2ints(train_file, train_file + "_ints", True, dict_dict)
            #test_length = utils.convert2ints(test_file, test_file + "_ints", True, dict_dict)

            train_date = LogFile(train_file, ",", 0, 1000000, None, "Case")
            train_date.remove_attributes(["Anomaly"])
            test_date = LogFile(test_file, ",", 0, 1000000, None, "Case",
                                string_2_int=train_date.string_2_int, int_2_string=train_date.int_2_string)
            model = edbn.train(train_date)
            edbn.test(test_date, path + "Output_%i_%i.csv" % anoms_rates[i], model, "Anomaly", 1)

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
