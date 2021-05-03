# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:16:15 2019

@author: Manuel Camargo
"""
import json
import math
import os
import random

import jellyfish as jf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from RelatedMethods.Camargo.support_modules import support as sup

START_TIMEFORMAT = ''
INDEX_AC = None
INDEX_RL = None
DIM = dict()
TBTW = dict()
EXP = dict()


def predict_next_old(timeformat, parameters, is_single_exec=True):
    """Main function of the suffix prediction module.
    Args:
        timeformat (str): event-log date-time format.
        parameters (dict): parameters used in the training step.
        is_single_exec (boolean): generate measurments stand alone or share
                    results with other runing experiments (optional)
    """
    global START_TIMEFORMAT
    global INDEX_AC
    global INDEX_RL
    global DIM
    global TBTW
    global EXP

    START_TIMEFORMAT = timeformat

    output_route = os.path.join('output_files', parameters['folder'])
    model_name, _ = os.path.splitext(parameters['model_file'])
    # Loading of testing dataframe
    df_test = pd.read_csv(os.path.join(output_route, 'parameters', 'test_log.csv'))
    df_test['start_timestamp'] = pd.to_datetime(df_test['start_timestamp'])
    df_test['end_timestamp'] = pd.to_datetime(df_test['end_timestamp'])
    df_test = df_test.drop(columns=['user'])
    df_test = df_test.rename(index=str, columns={"role": "user"})

    # Loading of parameters from training
    with open(os.path.join(output_route, 'parameters', 'model_parameters.json')) as file:
        data = json.load(file)
        EXP = {k: v for k, v in data['exp_desc'].items()}
        print(EXP)
        DIM['samples'] = int(data['dim']['samples'])
        DIM['time_dim'] = int(data['dim']['time_dim'])
        DIM['features'] = int(data['dim']['features'])
        TBTW['max_tbtw'] = float(data['max_tbtw'])
        INDEX_AC = {int(k): v for k, v in data['index_ac'].items()}
        INDEX_RL = {int(k): v for k, v in data['index_rl'].items()}
        file.close()

    if EXP['norm_method'] == 'max':
        max_tbtw = np.max(df_test.tbtw)
        norm = lambda x: x['tbtw'] / max_tbtw
        df_test['tbtw_norm'] = df_test.apply(norm, axis=1)
    elif EXP['norm_method'] == 'lognorm':
        logit = lambda x: math.log1p(x['tbtw'])
        df_test['tbtw_log'] = df_test.apply(logit, axis=1)
        max_tbtw = np.max(df_test.tbtw_log)
        norm = lambda x: x['tbtw_log'] / max_tbtw
        df_test['tbtw_norm'] = df_test.apply(norm, axis=1)

    ac_alias = create_alias(len(INDEX_AC))
    rl_alias = create_alias(len(INDEX_RL))

    #   Next event selection method and numbers of repetitions
    variants = [{'imp': 'Random Choice', 'rep': 15},
                {'imp': 'Arg Max', 'rep': 1}]
    #   Generation of predictions
    model = load_model(os.path.join(output_route, parameters['model_file']))

    for var in variants:
        measurements = list()
        for i in range(0, var['rep']):

            prefixes = create_pref_suf(df_test, ac_alias, rl_alias)
            prefixes = predict(model, prefixes, ac_alias, rl_alias, var['imp'])

            accuracy = (np.sum([x['ac_true'] for x in prefixes]) / len(prefixes))

            if is_single_exec:
                sup.create_csv_file_header(prefixes, os.path.join(output_route,
                                                                  model_name + '_rep_' + str(i) + '_next.csv'))

            # Save results
            measurements.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                        implementation=var['imp']), **{'accuracy': accuracy},
                                 **EXP})
        if measurements:
            if is_single_exec:
                sup.create_csv_file_header(measurements, os.path.join(output_route,
                                                                      model_name + '_next.csv'))
            else:
                if os.path.exists(os.path.join('output_files', 'next_event_measures.csv')):
                    sup.create_csv_file(measurements, os.path.join('output_files',
                                                                   'next_event_measures.csv'), mode='a')
                else:
                    sup.create_csv_file_header(measurements, os.path.join('output_files',
                                                                          'next_event_measures.csv'))


def predict_next(log, model):
    """Main function of the suffix prediction module.
    Args:
        timeformat (str): event-log date-time format.
        parameters (dict): parameters used in the training step.
        is_single_exec (boolean): generate measurments stand alone or share
                    results with other runing experiments (optional)
    """

    prefixes = create_pref_suf(log)
    return predict(model, prefixes)
            

# =============================================================================
# Predict traces
# =============================================================================

def predict(model, prefixes):
    """Generate business process suffixes using a keras trained model.
    Args:
        model (keras model): keras trained model.
        prefixes (list): list of prefixes.
        imp (str): method of next event selection.
    """
    # Generation of predictions
    results = []
    for prefix in prefixes:

        x_ac_ngram = np.array([prefix['ac_pref']])
        x_rl_ngram = np.array([prefix['rl_pref']])

        predictions = model.predict([x_ac_ngram, x_rl_ngram])

        pos = np.argmax(predictions[0][0])
        # pos = np.random.choice(np.arange(0, len(predictions[0][0])), p=predictions[0][0])

        results.append((prefix["ac_next"], pos, predictions[0][0][pos], predictions[0][0][prefix["ac_next"]]))

    return results

    #     # Roles accuracy evaluation
    #     if pos1 == prefix['rl_next']:
    #         prefix['rl_true'] = 1
    #     else:
    #         prefix['rl_true'] = 0
    # sup.print_done_task()
    # return prefixes


# =============================================================================
# Reformat
# =============================================================================
def create_pref_suf(log):
    """Extraction of prefixes and expected suffixes from event log.
    Args:
        df_test (dataframe): testing dataframe in pandas format.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        pref_size (int): size of the prefixes to extract.
    Returns:
        list: list of prefixes and expected sufixes.
    """
    prefixes = []
    cases = log.get_cases()
    for case in cases:
        trace = case[1]

        for row in trace.iterrows():
            row = row[1]
            ac_pref = []
            rl_pref = []
            t_pref = []
            for i in range(log.k - 1, -1, -1):
                ac_pref.append(row["event_Prev%i" % i])
                rl_pref.append(row["role_Prev%i" % i])
                t_pref.append(0)
            prefixes.append(dict(ac_pref=ac_pref,
                                 ac_next=row["event"],
                                 rl_pref=rl_pref,
                                 rl_next=row["role"],
                                 t_pref=t_pref))
    return prefixes


def create_alias(quantity):
    """Creates char aliases for a categorical attributes.
    Args:
        quantity (int): number of aliases to create.
    Returns:
        dict: alias for a categorical attributes.
    """
    characters = [chr(i) for i in range(0, quantity)]
    aliases = random.sample(characters, quantity)
    alias = dict()
    for i in range(0, quantity):
        alias[i] = aliases[i]
    return alias

def dl_measure(prefixes, feature):
    """Demerau-Levinstain distance measurement.
    Args:
        prefixes (list): list with predicted and expected suffixes.
        feature (str): categorical attribute to measure.
    Returns:
        list: list with measures added.
    """
    for prefix in prefixes:
        length = np.max([len(prefix[feature + '_suf']), len(prefix[feature + '_suf_pred'])])
        sim = jf.damerau_levenshtein_distance(prefix[feature + '_suf'],
                                              prefix[feature + '_suf_pred'])
        sim = (1-(sim/length))
        prefix[feature + '_dl'] = sim
    return prefixes

def ae_measure(prefixes):
    """Absolute Error measurement.
    Args:
        prefixes (list): list with predicted remaining-times and expected ones.
    Returns:
        list: list with measures added.
    """
    for prefix in prefixes:
        prefix['ae'] = abs(prefix['rem_time'] - prefix['rem_time_pred'])
    return prefixes
