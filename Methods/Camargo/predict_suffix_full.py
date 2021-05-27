# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:16:15 2019

@author: Manuel Camargo
"""
import os
import random

import numpy as np
import pandas as pd

from Methods.Camargo.support_modules import support as sup

START_TIMEFORMAT = ''
INDEX_AC = None
INDEX_RL = None
DIM = dict()
TBTW = dict()
EXP = dict()

def predict_suffix_full_old(model, data, is_single_exec=True):
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

    # Loading of testing dataframe
    df_test = pd.read_csv(os.path.join(data_folder, 'test_log.csv'))

    ac_index = {v: int(k) for k, v in data['index_ac'].items()}
    rl_index = {v: int(k) for k, v in data['index_rl'].items()}

    ac_alias = create_alias(len(INDEX_AC))
    rl_alias = create_alias(len(INDEX_RL))

    measurements_ac = list()
    measurements_rl = list()

    prefixes = create_pref_suf(df_test, ac_index, rl_index)
    prefixes = predict(model, prefixes, "Arg Max", 100)
    prefixes = dl_measure(prefixes, 'ac', ac_alias)
    prefixes = dl_measure(prefixes, 'rl', rl_alias)
    prefixes = pd.DataFrame.from_dict(prefixes)
    prefixes = prefixes.groupby('pref_size', as_index=False).agg({'ac_dl': 'mean','rl_dl': 'mean'})
    measure_ac = dict()
    measure_rl = dict()
    for size in prefixes.pref_size.unique():
        measure_ac[size] = prefixes[prefixes.pref_size==size].ac_dl.iloc[0]
        measure_rl[size] = prefixes[prefixes.pref_size==size].rl_dl.iloc[0]
    measure_ac['avg'] = prefixes.ac_dl.mean()
    measure_rl['avg'] = prefixes.rl_dl.mean()
    # Save results
    measurements_ac.append({**dict(model=os.path.join(output_route, model_file),
                                implementation=var['imp']), **measure_ac,
                        **EXP})
    measurements_rl.append({**dict(model=os.path.join(output_route, model_file),
                                implementation=var['imp']), **measure_rl,
                        **EXP})


    return measure_ac['avg']

def predict_suffix(model, log):
    """Main function of the suffix prediction module.
    Args:
        timeformat (str): event-log date-time format.
        parameters (dict): parameters used in the training step.
        is_single_exec (boolean): generate measurments stand alone or share
                    results with other runing experiments (optional)
    """

    prefixes = create_pref_suf(log.test_orig)
    prefixes = predict(model, prefixes, log.logfile.convert_string2int(log.logfile.activity, "end"))
    prefixes = dl_measure(prefixes, 'ac')
    return prefixes.ac_dl.mean()

def save_results(measurements, feature, is_single_exec, model_file, output_folder):
    model_name, _ = os.path.splitext(model_file)
    if measurements:    
        if is_single_exec:
                sup.create_csv_file_header(measurements, os.path.join(output_folder,
                                                                      model_name +'_'+feature+'_full_suff.csv'))
        else:
            if os.path.exists(os.path.join(output_folder, 'full_'+feature+'_suffix_measures.csv')):
                sup.create_csv_file(measurements, os.path.join(output_folder,
                                                               'full_'+feature+'_suffix_measures.csv'), mode='a')
            else:
                sup.create_csv_file_header(measurements, os.path.join(output_folder,
                                                               'full_'+feature+'_suffix_measures.csv'))

# =============================================================================
# Predic traces
# =============================================================================

def predict(model, prefixes, end_index):
    results = []
    for prefix in prefixes:

        x_ac_ngram = np.array([prefix['ac_pref']])
        x_rl_ngram = np.array([prefix['rl_pref']])

        max_trace_size = 100
        ac_suf, rl_suf = list(), list()
        for _  in range(1, max_trace_size):
            predictions = model.predict([x_ac_ngram, x_rl_ngram])

            pos = np.argmax(predictions[0][0])
            pos1 = np.argmax(predictions[1][0])

            # pos = np.random.choice(np.arange(0, len(predictions[0][0])), p=predictions[0][0])
            # Activities accuracy evaluation
            x_ac_ngram = np.append(x_ac_ngram, [[pos]], axis=1)
            x_ac_ngram = np.delete(x_ac_ngram, 0, 1)
            x_rl_ngram = np.append(x_rl_ngram, [[pos1]], axis=1)
            x_rl_ngram = np.delete(x_rl_ngram, 0, 1)
            # Stop if the next prediction is the end of the trace
            # otherwise until the defined max_size
            ac_suf.append(pos)
            rl_suf.append(pos1)

            if pos == end_index:
                break

        prefix['ac_suff_pred'] = ac_suf
        prefix['rl_suff_pred'] = rl_suf

    return prefix

def predict_old(model, prefixes, imp="", max_trace_size=100):
    """Generate business process suffixes using a keras trained model.
    Args:
        model (keras model): keras trained model.
        prefixes (list): list of prefixes.
        imp (str): method of next event selection.
    """
    # Generation of predictions
    for prefix in prefixes:
        # Activities and roles input shape(1,5)
        x_ac_ngram = np.append(
                np.zeros(DIM['time_dim']),
                np.array(prefix['ac_pref']),
                axis=0)[-DIM['time_dim']:].reshape((1,DIM['time_dim']))
                
        x_rl_ngram = np.append(
                np.zeros(DIM['time_dim']),
                np.array(prefix['rl_pref']),
                axis=0)[-DIM['time_dim']:].reshape((1,DIM['time_dim']))

        ac_suf, rl_suf = list(), list()
        for _  in range(1, max_trace_size):
            predictions = model.predict([x_ac_ngram, x_rl_ngram])
            if imp == 'Random Choice':
                # Use this to get a random choice following as PDF the predictions
                pos = np.random.choice(np.arange(0, len(predictions[0][0])), p=predictions[0][0])
                pos1 = np.random.choice(np.arange(0, len(predictions[1][0])), p=predictions[1][0])
            elif imp == 'Arg Max':
                # Use this to get the max prediction
                pos = np.argmax(predictions[0][0])
                pos1 = np.argmax(predictions[1][0])
            # Activities accuracy evaluation
            x_ac_ngram = np.append(x_ac_ngram, [[pos]], axis=1)
            x_ac_ngram = np.delete(x_ac_ngram, 0, 1)
            x_rl_ngram = np.append(x_rl_ngram, [[pos1]], axis=1)
            x_rl_ngram = np.delete(x_rl_ngram, 0, 1)
            # Stop if the next prediction is the end of the trace
            # otherwise until the defined max_size
            ac_suf.append(pos)
            rl_suf.append(pos1)

            if INDEX_AC[pos] == 'end':
                break

        prefix['ac_suff_pred'] = ac_suf
        prefix['rl_suff_pred'] = rl_suf
    sup.print_done_task()
    return prefixes


# =============================================================================
# Reformat
# =============================================================================

def create_pref_suf(log):
    prefixes = []
    cases = log.get_cases()
    for case in cases:
        trace = case[1]

        trace_ac = list(trace["event"])
        trace_rl = list(trace["role"])

        j = 0
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
                                 ac_suff=[x for x in trace_ac[j + 1:]],
                                 rl_pref=rl_pref,
                                 rl_suff=[x for x in trace_rl[j + 1:]],
                                 t_pref=t_pref))
            j += 1
    return prefixes



def create_pref_suf_old(df_test, ac_index, rl_index):
    """Extraction of prefixes and expected suffixes from event log.
    Args:
        df_test (dataframe): testing dataframe in pandas format.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        pref_size (int): size of the prefixes to extract.
    Returns:
        list: list of prefixes and expected sufixes.
    """
    prefixes = list()
    cases = df_test.case.unique()
    for case in cases:
        trace = df_test[df_test.case == case].to_dict('records')
        ac_pref = list()
        rl_pref = list()
        for i in range(0, len(trace)-1):
            ac_pref.append(trace[i]['event'])
            rl_pref.append(trace[i]['role'])
            prefixes.append(dict(ac_pref=ac_pref.copy(),
                                 ac_suff=[x['event'] for x in trace[i + 1:]],
                                 rl_pref=rl_pref.copy(),
                                 rl_suff=[x['role'] for x in trace[i + 1:]],
                                 pref_size=i + 1))
    for x in prefixes:
        x['ac_suff'].append(ac_index['end'])
        x['rl_suff'].append(rl_index['end'])
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
        length = np.max([len(prefix[feature + '_suff']), len(prefix[feature + '_suff_pred'])])
        sim = damerau_levenshtein_distance(prefix[feature + '_suff'], prefix[feature + '_suff_pred'])
        sim = (1-(sim/length))
        prefix[feature + '_dl'] = sim
    return prefixes

"""
Compute the Damerau-Levenshtein distance between two given
lists (s1 and s2)
From: https://www.guyrutenberg.com/2008/12/15/damerau-levenshtein-distance-in-python/
"""
def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]

def ae_measure(prefixes):
    """Absolute Error measurement.
    Args:
        prefixes (list): list with predicted remaining-times and expected ones.
    Returns:
        list: list with measures added.
    """
    for prefix in prefixes:
        rem_log = np.sum(prefix['rem_time'])
#        prefix['ae'] = abs(prefix['rem_time'] - prefix['rem_time_pred'])
        prefix['ae'] = abs(rem_log - prefix['rem_time_pred'])
    return prefixes
