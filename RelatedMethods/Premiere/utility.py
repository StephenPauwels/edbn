import datetime
import numpy as np
from datetime import datetime
from nltk import ngrams
import pandas as pd
import time

def get_size_fold(namedataset):
    fold1 = pd.read_csv('fold/' + namedataset + '_premiereFold0' + '.txt', header=None)
    fold2 = pd.read_csv('fold/' + namedataset + '_premiereFold1' + '.txt', header=None)
    fold3 = pd.read_csv('fold/' + namedataset + '_premiereFold2' + '.txt', header=None)

    fold1.columns = ["CaseID", "Activity", "Resource", "Timestamp"]
    fold2.columns = ["CaseID", "Activity", "Resource", "Timestamp"]
    fold3.columns = ["CaseID", "Activity", "Resource", "Timestamp"]

    n_caseid1 = fold1['CaseID'].nunique()
    n_caseid2 = fold2['CaseID'].nunique()
    n_caseid3 = fold3['CaseID'].nunique()

    len_fold1 = len(fold1)
    len_fold2 = len(fold2)
    len_fold3 = len(fold3)

    print("n_caseid1",n_caseid1)
    print("n_caseid2",n_caseid2)
    print("n_caseid3",n_caseid3)

    nos1 = len_fold1 - n_caseid1
    nos2 = len_fold2 - n_caseid2
    nos3 = len_fold3 - n_caseid3

    return nos1, nos2, nos3

def get_label(prefix,max_trace):
    i = 0
    s = (max_trace)
    list_seq = []
    list_label = []
    while i < len(prefix):
        list_temp = []
        seq = np.zeros(s)
        j = 0
        while j < (len(prefix.iat[i, 0]) - 1):
            list_temp.append(prefix.iat[i, 0][0 + j])
            new_seq = np.append(seq, list_temp)
            cut = len(list_temp)
            new_seq = new_seq[cut:]
            list_seq.append(new_seq)
            list_label.append(prefix.iat[i, 0][j + 1])
            j = j + 1
        i = i + 1
    return list_label


def res_feature(resource_prefix, resource_list):
    resource_features = []
    for res in resource_list:
        resource_features.append(resource_prefix.count(res))
    return resource_features


def act_feature(sequence_prefix, activity_list):
    activity_features = []
    for act in activity_list:
        activity_features.append(sequence_prefix.count(act))
    return activity_features


def flow_feature(sequence_prefix, flow_act):
    return_list = [0] * len(flow_act)
    for gram in ngrams(sequence_prefix, 2):
        return_list[flow_act[gram]] += 1
    return return_list

    # return [list_gram.count(fl_act) for fl_act in flow_act]


def combine(ran, flow, activity, resource, agg_time, target):
    result = []
    for i in ran:
        result.append(flow[i] + activity[i] + resource[i] + agg_time[i] + [target[i]])
    return result


def calc_features(ran, list_sequence_prefix, list_resource_prefix, activity_list, resource_list, flow_act, agg_time, target):
    output = pd.DataFrame()
    for i in ran:
        output = output.append([flow_feature(list_sequence_prefix[i], flow_act) +
                               act_feature(list_sequence_prefix[i], activity_list) +
                               res_feature(list_resource_prefix[i], resource_list) +
                               agg_time[i] + [target[i]]])
    return output


def premiere_feature(list_sequence_prefix, list_resource_prefix, flow_act, agg_time_feature, unique_events, unique_resources, target):
    print(len(list_sequence_prefix), len(agg_time_feature), len(target))
    j = 0
    list_flow_feature = []
    n_resource_list = list(range(1, unique_resources + 1))
    n_activity_list = list(range(1, unique_events + 1))

    import multiprocessing
    import functools
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    # print("Resource features")
    # resource_features = pool.map(functools.partial(res_feature, resource_list=n_resource_list), list_resource_prefix)
    # print("Activity features")
    # activity_features = pool.map(functools.partial(act_feature, activity_list=n_activity_list), list_sequence_prefix)
    # print("Flow features")
    flow_act_dict = {flow: i for i, flow in enumerate(flow_act)}
    # flow_features = pool.map(functools.partial(flow_feature, flow_act=flow_act_dict), list_sequence_prefix)

    # print("Combine", len(list_sequence_prefix))
    chunk_size = int(len(list_sequence_prefix) / num_processes)
    chunks = [range(i, min(i + chunk_size, len(list_sequence_prefix))) for i in range(0, len(list_sequence_prefix), chunk_size)]
    print(chunks[-1])
    # list_flow_feature = []
    # result = pool.map(functools.partial(combine, flow=flow_features, activity=activity_features,
    #                                                resource=resource_features, agg_time=agg_time_feature, target=target)
    #                              , chunks)

    result = pool.map(functools.partial(calc_features, list_sequence_prefix=list_sequence_prefix,
                                        list_resource_prefix=list_resource_prefix, activity_list=n_activity_list,
                                        resource_list=n_resource_list, flow_act=flow_act_dict, agg_time=agg_time_feature,
                                        target=target), chunks)

    # for j in range(len(list_sequence_prefix)):
    #     list_flow_feature.append(flow_features[j] + activity_features[j] + resource_features[j] + agg_time_feature[j] + [target[j]])
    print("Done Combine")

    return pd.concat(result)


def output_list(masterList):
    output = []
    for item in masterList:
        if isinstance(item, list):  # if item is a list
            for i in output_list(item):  # call this function on it and append its each value separately. If it has more lists in it this function will call itself again
                output.append(i)
        else:
            output.append(item)
    return output


def get_time(prova,max_trace):
    i = 0
    s = (max_trace)
    list_seq = []
    datetimeFormat = '%Y/%m/%d %H:%M:%S.%f'
    while i < len(prova):
        list_temp = []
        seq = np.zeros(s)
        j = 0
        while j < (len(prova.iat[i, 0]) - 1):
            t = time.strptime(prova.iat[i, 0][0 + j], datetimeFormat)
            list_temp.append(datetime.fromtimestamp(time.mktime(t)))
            new_seq = np.append(seq, list_temp)
            cut = len(list_temp)
            new_seq = new_seq[cut:]
            list_seq.append(new_seq)
            j = j + 1
        i = i + 1
    return list_seq


def get_sequence(prefix,max_trace):
    i = 0
    s = (max_trace)
    list_seq = []
    list_label = []
    while i < len(prefix):
        list_temp = []
        seq = np.zeros(s)
        j = 0
        while j < (len(prefix.iat[i, 0]) - 1):
            list_temp.append(prefix.iat[i, 0][0 + j])
            new_seq = np.append(seq, list_temp)
            cut = len(list_temp)
            new_seq = new_seq[cut:]
            new_seq = new_seq.astype(int)
            list_seq.append(new_seq)
            list_label.append(prefix.iat[i, 0][j + 1])
            j = j + 1
        i = i + 1
    return list_seq