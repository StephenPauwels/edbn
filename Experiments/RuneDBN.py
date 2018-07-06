#import eDBN.Execute as edbn
import Utils.Utils as utils

import pandas as pd
import extended_Dynamic_Bayesian_Network as edbn

if __name__ == "__main__":
    file = "../Data/bpic2018_ints.csv"

    #output_edbn = "../Data/Output_2/BPIC15_edbn_output_1.csv"
    #ignore_attrs = ["eventid", "identity_id", "event_identity_id", "year", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"]

    #edbn_model = edbn.train(file, "case", "Anomaly", 1, header=0, length=30000, ignore=ignore_attrs)
    #edbn.test(file, output_edbn, edbn_model, "Anomaly", 1, ",", length=100000, skip=0)

    data = pd.read_csv(file, delimiter=",", nrows=100000000, header=0, dtype=int, skiprows=0)
    model = edbn.extendedDynamicBayesianNetwork(len(data.columns), 1, "case", "Anomaly", 1)
    context = model.create_k_context(data)

    print(context)