import eDBN.Execute as edbn
import Utils.DataGenerator as generator
import Utils.PlotResults as plt
import Utils.Utils as utils

if __name__ == "__main__":
    path = "../Data/Experiments/"
    train_rates = [0] #[0,5,10,25]
    test_rates = [5] #[1,5,10,25,50,100,250,500,750]
    anoms_rates = []
    for train_rate in train_rates:
        for test_rate in test_rates:
            anoms_rates.append((train_rate, test_rate))

    for i in range(len(anoms_rates)) :
        train_file = path + "%i_train_%i.csv" %(i, anoms_rates[i][0])
        test_file = path + "%i_test_%i.csv" %(i, anoms_rates[i][1])
        #generator.create_shipment_data(10000, 10000, anoms_rates[i][0], anoms_rates[i][1], train_file, test_file)

        dict_dict = []
        utils.convert2ints(train_file, train_file + "_ints", True, dict_dict)
        test_length = utils.convert2ints(test_file, test_file + "_ints", True, dict_dict)

        model = edbn.train(test_file + "_ints", "Case", "Anomaly", 1, 0, 1000000, ignore=["Anomaly"])
        edbn.test(test_file + "_ints", path + "Output_%i_%i.csv" % anoms_rates[i], model, "Anomaly", 1, ",", test_length, skip=0)

        output_file = path + "Output_%i_%i.csv" % anoms_rates[i]
        output_roc = path + "roc_%i_%i.png" % anoms_rates[i]
        output_prec = path + "prec_recall_%i_%i.png" % anoms_rates[i]
        plt.plot_single_roc_curve(output_file, output_roc)
        plt.plot_single_prec_recall_curve(output_file, None, output_prec)
