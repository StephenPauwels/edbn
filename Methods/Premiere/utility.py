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
    filename = "features/features_%i.csv" % ran.start
    with open(filename, "w") as fout:
        for i in ran:
            fout.write(",".join([str(j) for j in flow_feature(list_sequence_prefix[i], flow_act) +
                                 act_feature(list_sequence_prefix[i], activity_list) +
                                 res_feature(list_resource_prefix[i], resource_list) +
                                 agg_time[i] + [target[i]]]))
            fout.write("\n")
    return filename


def premiere_feature(list_sequence_prefix, list_resource_prefix, flow_act, agg_time_feature, unique_events, unique_resources, target, file=""):
    j = 0
    list_flow_feature = []
    n_resource_list = list(range(1, unique_resources + 1))
    n_activity_list = list(range(1, unique_events + 1))

    output_file = open(file, "w")

    while j < len(list_sequence_prefix):
        print(j, "/", len(list_sequence_prefix))
        activity_features = []
        resource_features = []

        # i = 0
        # while i < len(n_resource_list):
        tmp = list_resource_prefix[j]
        cnt = {}
        for i in tmp:
            cnt[i] = cnt.get(i, 0) + 1
        for i in n_resource_list:
            resource_features.append(cnt.get(i,0))
            # i = i + 1

        # x = 0
        # while x < len(n_activity_list):
        tmp = list_sequence_prefix[j]
        cnt = {}
        for x in tmp:
            cnt[x] = cnt.get(x, 0) + 1
        for x in n_activity_list:
            activity_features.append(cnt.get(x,0))
            # x = x + 1

        k = 0
        list_gram = list(ngrams(list_sequence_prefix[j], 2))

        # for grams in bigrams:
        #     list_gram.append(grams)

        flow_feature = []
        # while k < len(flow_act):
        cnt = {}
        for k in list_gram:
            cnt[k] = cnt.get(k, 0) + 1
        for k in flow_act:
            # print("Sequence->",a[k]," find n. ", list_gram.count(a[k]))
            flow_feature.append(cnt.get(k, 0))
            # k = k + 1

        # list_flow_feature.append(flow_feature + activity_features + resource_features + agg_time_feature[j] + [target[j]])
        #list_flow_feature.append(activity_features + resource_features + agg_time_feature[j] + [target[j]])

        output_file.write(",".join([str(j) for j in flow_feature + activity_features + resource_features +
                                    agg_time_feature[j] + [target[j]]]))
        output_file.write("\n")
        if j % 1000 == 0:
            output_file.flush()

        del activity_features
        del resource_features
        del flow_feature
        j = j + 1
    output_file.close()
    # return list_flow_feature


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