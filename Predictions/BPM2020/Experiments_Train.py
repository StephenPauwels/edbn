"""
    Basic file to train the models for the prediction experiments
    Use: python Experiments_Train METHOD DATA [args]

    Valid values for METHOD and DATA can be found in Experiments_Variables.py

    Extra arguments:
        - EDBN: * first extra argument: k-value
                * second extra argument: use "next" when training optimized for next event, use "suffix" for suffix
        - CAMARGO: specify architecture to use: chared_cat or specialized

    Author: Stephen Pauwels
"""

import os
import pickle
import sys
import time

OUTPUT_FOLDER = "Output/"


def main(argv):
    from Predictions.setting import Setting
    from Methods import get_prediction_method
    from Data import get_data

    if len(argv) < 2:
        print("Missing arguments, expected: METHOD and DATA")
        return

    method = argv[0]
    data = argv[1]

    ###
    # Load data, setting and method
    ###
    basic_setting = Setting(None, "test-train", False, True, 70, filter_cases=5)
    m = get_prediction_method(method)
    d = get_data(data)

    model_folder = os.path.join(OUTPUT_FOLDER, str.lower(d.name), "models", str.lower(method))

    ###
    # Check if all required folders exist
    ###
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    if not os.path.exists(os.path.join(OUTPUT_FOLDER, str.lower(d.name))):
        os.mkdir(os.path.join(OUTPUT_FOLDER, str.lower(d.name)))

    if not os.path.exists(os.path.join(OUTPUT_FOLDER, str.lower(d.name), "models")):
        os.mkdir(os.path.join(OUTPUT_FOLDER, str.lower(d.name), "models"))

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    # Perform some method specific checks
    if method == "DBN":
        if len(argv) >= 3:
            basic_setting.prefixsize = int(argv[2])
        else:
            basic_setting.prefixsize = 2
    elif method == "CAMARGO":
        if len(argv) < 3:
            print("Please indicate the architecture to use: shared_cat or specialized")
            return
        architecture = str.lower(argv[2])
        m.def_params["model_type"] = architecture
        basic_setting.prefixsize = 5
    elif method == "LIN":
        basic_setting.prefixsize = 5

    # Perform some data specific checks
    if data == "Helpdesk":
        basic_setting.filter_cases = 3

    d.prepare(basic_setting)

    ###
    # Register Start time
    ###
    start_time = time.mktime(time.localtime())
    start_time_str = time.strftime("%d-%m-%y %H:%M:%S", time.localtime())
    time_output = open(os.path.join(model_folder, "timings_train.log"), 'a')
    time_output.write("Starting time: %s\n" % start_time_str)

    ###
    # Train and save model using chosen data and method
    ###
    print("EXPERIMENT TRAINING MODEL:", argv)
    # Train model
    model = m.train(d.train)

    # Save model
    if method == "DBN":
        with open(os.path.join(model_folder, "model"), "wb") as pickle_file:
            pickle.dump(model, pickle_file)
    else:
        model.save(os.path.join(model_folder, "model.h5"))

    ###
    # Register End time
    ###
    current_time = time.mktime(time.localtime())
    current_time_str = time.strftime("%d-%m-%y %H:%M:%S", time.localtime())
    time_output.write("End time: %s\n" % current_time_str)
    time_output.write("Duration: %fs\n\n" % (current_time - start_time))


if __name__ == "__main__":
    main(sys.argv[1:])

