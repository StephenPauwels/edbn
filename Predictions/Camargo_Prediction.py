import sys

import Preprocessing as data

import Camargo.embedding_training as em
import Camargo.model_training as mo

def main(argv):
    dataset = argv[0]
    dataset_size = 200000
    add_end = True
    resource_pools = False
    reduce_tasks = False
    logfile_k = 2
    bpic_file = 5

    remove_resource = False

    logfile = data.get_data(dataset, dataset_size, logfile_k, add_end, reduce_tasks, resource_pools, remove_resource)
    logfile_df = logfile.data
    logfile_df.columns = ["caseid", "task", "role"]

    args = {}
    args["file_name"] = dataset
    args["model_type"] = argv[1] # Choose from 'joint', 'shared', 'concatenated', 'specialized', 'shared_cat'
    args["norm_method"] = "lognorm" # Choose from 'lognorm' or 'max'
    args["n_size"] = 5 # n-gram size
    args['lstm_act'] = None # optimization function see keras doc
    args['l_size'] = 100 # LSTM layer sizes
    args['imp'] = 1 # keras lstm implementation 1 cpu, 2 gpu
    args['dense_act'] = None # optimization function see keras doc
    args['optim'] = 'Nadam' # optimization function see keras doc

    em.training_model(logfile.data, dataset)
    mo.training_model(logfile.data, dataset, args)

if __name__ == "__main__":
    main(sys.argv[1:])