# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:23:55 2018

@author: Manuel Camargo
"""
import csv
import itertools
import multiprocessing as mp
from functools import partial

import tensorflow.keras.utils as ku
import numpy as np
import pandas as pd
from nltk.util import ngrams

from Methods.Camargo.models import model_shared_cat as mshcat
from Methods.Camargo.models import model_specialized as msp


def training_model(log, event_emb, role_emb, args, epochs, early_stop):
    """Main method of the training module.
    """

    # Load embedded matrix
    ac_weights = np.array(event_emb)
    rl_weights = np.array(role_emb)
    # Calculate relative times
    # log_df_train = add_calculated_features(log.contextdata)

    vec = vectorization(log)
    # Parameters export
    # output_folder = outfile

    parameters = {}
    parameters['event_log'] = args['file_name']
    parameters['exp_desc'] = args
    parameters['dim'] = dict(samples=str(vec['prefixes']['x_ac_inp'].shape[0]),
                             time_dim=str(vec['prefixes']['x_ac_inp'].shape[1]),
                             features=str(len(event_emb)))

    model = None
    if args['model_type'] == 'specialized':
        model = msp.training_model(vec, ac_weights, rl_weights, "tmp", args, epochs, early_stop)
    elif args['model_type'] == 'shared_cat':
        model = mshcat.training_model(vec, ac_weights, rl_weights, "tmp", args, epochs, early_stop)

    return model

# =============================================================================
# Load embedded matrix
# =============================================================================

def load_embedded(index, filename):
    """Loading of the embedded matrices.
    Args:
        index (dict): index of activities or roles.
        filename (str): filename of the matrix file.
    Returns:
        numpy array: array of weights.
    """
    weights = list()
    with open(filename, 'r') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in filereader:
            cat_ix = int(row[0])
            if str(index[cat_ix]) == row[1].strip():
                weights.append([float(x) for x in row[2:]])
        csvfile.close()
    return np.array(weights)

# =============================================================================
# Pre-processing: n-gram vectorization
# =============================================================================
def vectorization(log):
    """Example function with types documented in the docstring.
    Args:
        log_df (dataframe): event log data.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
    Returns:
        dict: Dictionary that contains all the LSTM inputs.
    """
    print("Start Vectorization")

    vec = {'prefixes': dict(), 'next_evt': dict()}

    train_cases = log.get_cases()
    part_vect_map = partial(vect_map, prefix_size=log.k)
    with mp.Pool(mp.cpu_count()) as p:
        result = np.array(p.map(part_vect_map, train_cases))

    vec['prefixes']['x_ac_inp'] = np.concatenate(result[:, 0])
    vec['prefixes']['x_rl_inp'] = np.concatenate(result[:, 1])
    vec['next_evt']['y_ac_inp'] = np.concatenate(result[:, 2])
    vec['next_evt']['y_rl_inp'] = np.concatenate(result[:, 3])

    vec['next_evt']['y_ac_inp'] = ku.to_categorical(vec['next_evt']['y_ac_inp'], num_classes=len(log.values["event"])+1)
    vec['next_evt']['y_rl_inp'] = ku.to_categorical(vec['next_evt']['y_rl_inp'], num_classes=len(log.values["role"])+1)
    return vec


def map_case(x, log_df, case_attr):
    return log_df[log_df[case_attr] == x]


def vect_map(case, prefix_size):
    case_df = case[1]

    x_ac_inps = []
    x_rl_inps = []
    y_ac_inps = []
    y_rl_inps = []
    for row in case_df.iterrows():
        row = row[1]
        x_ac_inp = []
        x_rl_inp = []
        for i in range(prefix_size - 1, 0, -1):
            x_ac_inp.append(row["event_Prev%i" % i])
            x_rl_inp.append(row["role_Prev%i" % i])
        x_ac_inp.append(row["event_Prev0"])
        x_rl_inp.append(row["role_Prev0"])

        x_ac_inps.append(x_ac_inp)
        x_rl_inps.append(x_rl_inp)
        y_ac_inps.append(row["event"])
        y_rl_inps.append(row["role"])
    return [np.array(x_ac_inps), np.array(x_rl_inps), np.array(y_ac_inps), np.array(y_rl_inps)]


def add_calculated_features(log_df):
    """Appends the indexes and relative time to the dataframe.
    Args:
        log_df: dataframe.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
    Returns:
        Dataframe: The dataframe with the calculated features added.
    """
    # ac_idx = lambda x: ac_index[x['task']]
    # log_df['ac_index'] = log_df.apply(ac_idx, axis=1)
    log_df['ac_index'] = log_df['event']

    # rl_idx = lambda x: rl_index[x['role']]
    # log_df['rl_index'] = log_df.apply(rl_idx, axis=1)
    log_df['rl_index'] = log_df['role']

    log_df['tbtw'] = 0
    log_df['tbtw_norm'] = 0

    log_df = log_df.to_dict('records')

#    log_df = sorted(log_df, key=lambda x: (x['caseid'], x['end_timestamp']))
#    for _, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
#        trace = list(group)
#        for i, _ in enumerate(trace):
#            if i != 0:
#                trace[i]['tbtw'] = (trace[i]['end_timestamp'] -
#                                    trace[i-1]['end_timestamp']).total_seconds()

    return pd.DataFrame.from_records(log_df)

def reformat_events(log_df, ac_index, rl_index):
    """Creates series of activities, roles and relative times per trace.
    Args:
        log_df: dataframe.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
    Returns:
        list: lists of activities, roles and relative times.
    """
    log_df = log_df.to_dict('records')
    print(rl_index)
    temp_data = list()
#    log_df = sorted(log_df, key=lambda x: (x['caseid'], x['end_timestamp']))
    for key, group in itertools.groupby(log_df, key=lambda x: x['case']):
        trace = list(group)
        ac_order = [x['ac_index'] for x in trace]
        rl_order = [x['rl_index'] for x in trace]
        tbtw = [x['tbtw_norm'] for x in trace]
        ac_order.insert(0, ac_index[('start')])
        ac_order.append(ac_index[('end')])
        rl_order.insert(0, rl_index[('start')])
        rl_order.append(rl_index[('end')])
        tbtw.insert(0, 0)
        tbtw.append(0)
        temp_dict = dict(caseid=key,
                         ac_order=ac_order,
                         rl_order=rl_order,
                         tbtw=tbtw)
        temp_data.append(temp_dict)

    return temp_data


# =============================================================================
# Support
# =============================================================================


def create_index(log_df, column):
    """Creates an idx for a categorical attribute.
    Args:
        log_df: dataframe.
        column: column name.
    Returns:
        index of a categorical attribute pairs.
    """
    temp_list = log_df[[column]].values.tolist()
    subsec_set = {(x[0]) for x in temp_list}
    subsec_set = sorted(list(subsec_set))
    alias = dict()
    for i, _ in enumerate(subsec_set):
        alias[subsec_set[i]] = i + 1
    return alias

def max_serie(log_df, serie):
    """Returns the max and min value of a column.
    Args:
        log_df: dataframe.
        serie: name of the serie.
    Returns:
        max and min value.
    """
    max_value, min_value = 0, 0
    for record in log_df:
        if np.max(record[serie]) > max_value:
            max_value = np.max(record[serie])
        if np.min(record[serie]) > min_value:
            min_value = np.min(record[serie])
    return max_value, min_value

def max_min_std(val, max_value, min_value):
    """Standardize a number between range.
    Args:
        val: Value to be standardized.
        max_value: Maximum value of the range.
        min_value: Minimum value of the range.
    Returns:
        Standardized value between 0 and 1.
    """
    std = (val - min_value) / (max_value - min_value)
    return std
