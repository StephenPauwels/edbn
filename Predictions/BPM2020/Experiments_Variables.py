"""
    Author: Stephen Pauwels
"""
EDBN = "EDBN"
CAMARGO = "CAMARGO"
DIMAURO = "DIMAURO"
LIN = "LIN"
TAX = "TAX"
METHODS = [EDBN, CAMARGO, DIMAURO, LIN, TAX]

DATA_DESC = []
DATA_DESC.append({"folder": "bpic15_1/", "data": "BPIC15_1"})
DATA_DESC.append({"folder": "bpic15_2/", "data": "BPIC15_2"})
DATA_DESC.append({"folder": "bpic15_3/", "data": "BPIC15_3"})
DATA_DESC.append({"folder": "bpic15_4/", "data": "BPIC15_4"})
DATA_DESC.append({"folder": "bpic15_5/", "data": "BPIC15_5"})
DATA_DESC.append({"folder": "bpic12w/", "data": "BPIC12W"})
DATA_DESC.append({"folder": "bpic12/", "data": "BPIC12"})
DATA_DESC.append({"folder": "helpdesk/", "data": "HELPDESK"})
# DATA_DESC.append({"folder": "bpic18/", "data": "BPIC18"})
# DATA_DESC.append({"folder": "bpic18s/", "data": "BPIC18S"})
# DATA_DESC.append({"folder": "bpic18m/", "data": "BPIC18M"})
# DATA_DESC.append({"folder": "bpic18l/", "data": "BPIC18L"})

# Location to output the splitted log files
DATA_FOLDER = "../../Data/PredictionData/"

# Location to output model and result files
OUTPUT_FOLDER = "Output/"

# K-value for EDBN
K_EDBN = 3

# Best Params for DIMAURO
DIMAURO_PARAMS = {}
DIMAURO_PARAMS["BPIC12"] = {'batch_size': 9, 'embedding_size': 12, 'input_length': 105,
                                          'learning_rate': 0.006114236931593503, 'model_type': 'ACT', 'n_classes': 22,
                                          'n_modules': 3, 'vocab_size': 23}
DIMAURO_PARAMS["BPIC12W"] = {'batch_size': 10, 'embedding_size': 3, 'input_length': 73,
                             'learning_rate': 0.002091242548380724, 'model_type': 'ACT', 'n_classes': 6,
                             'n_modules': 3, 'vocab_size': 6}
DIMAURO_PARAMS["BPIC15_1"] = {'batch_size': 9, 'embedding_size': 199, 'input_length': 100,
                              'learning_rate': 0.0008106393532786784, 'model_type': 'ACT', 'n_classes': 398,
                              'n_modules': 2, 'vocab_size': 398}
DIMAURO_PARAMS["BPIC15_2"] = {'batch_size': 9, 'embedding_size': 205, 'input_length': 131,
                              'learning_rate': 0.0029149847014206063, 'model_type': 'ACT', 'n_classes': 410,
                              'n_modules': 2, 'vocab_size': 410}
DIMAURO_PARAMS["BPIC15_3"] = {'batch_size': 10, 'embedding_size': 192, 'input_length': 123,
                              'learning_rate': 0.0025149353506156544, 'model_type': 'ACT', 'n_classes': 383,
                              'n_modules': 2, 'vocab_size': 383}
DIMAURO_PARAMS["BPIC15_4"] = {'batch_size': 9, 'embedding_size': 178, 'input_length': 115,
                              'learning_rate': 0.0008106393532786784, 'model_type': 'ACT', 'n_classes': 356,
                              'n_modules': 2, 'vocab_size': 356}
DIMAURO_PARAMS["BPIC15_5"] = {'batch_size': 10, 'embedding_size': 195, 'input_length': 153,
                              'learning_rate': 0.0005646310499409864, 'model_type': 'ACT', 'n_classes': 389,
                              'n_modules': 3, 'vocab_size': 389}
DIMAURO_PARAMS["HELPDESK"] = {'batch_size': 9, 'embedding_size': 7, 'input_length': 14,
                              'learning_rate': 0.0008106393532786784, 'model_type': 'ACT', 'n_classes': 13,
                              'n_modules': 2, 'vocab_size': 14}
DIMAURO_PARAMS["BPIC18"] = {'batch_size': 9, 'embedding_size': 21, 'input_length': 2973,
                              'learning_rate': 0.0008106393532786784, 'model_type': 'ACT', 'n_classes': 42,
                              'n_modules': 2, 'vocab_size': 42}