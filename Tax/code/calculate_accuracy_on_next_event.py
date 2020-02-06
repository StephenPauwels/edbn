'''
this script takes as input the output of evaluate_suffix_and_remaining_time.py
therefore, the latter needs to be executed first

Author: Niek Tax
'''

from __future__ import division
import unicodecsv
import csv

# CaseID,Prefix length,Groud truth,Predicted,Levenshtein,Damerau,Jaccard,Ground truth times,Predicted times,RMSE,MAE

#eventlogs = ["helpdesk", "bpic12", "bpic12w", "bpic15_1", "bpic15_2", "bpic15_3", "bpic15_4", "bpic15_5"]
eventlogs = ["BPIC15_1b"]

for eventlog_name in eventlogs:
    csvfile = open('output_files/results/next_activity_and_time_%s' % eventlog_name, 'r')
    r = csv.reader(csvfile)
    next(r,None) # header
    vals = dict()
    for row in r:
        l = list()
        if row[0] in vals.keys():
            l = vals.get(row[0])
        if len(row[2])==0 and len(row[3])==0:
            l.append(1)
        elif len(row[2])==0 and len(row[3])>0:
            l.append(0)
        elif len(row[2])>0 and len(row[3])==0:
            l.append(0)
        else:
            l.append(int(row[2][0]==row[3][0]))
        vals[row[0]] = l
        #print(vals)

    l2 = list()
    for k in vals.keys():
        #print('{}: {}'.format(k, vals[k]))
        l2.extend(vals[k])
        res = sum(vals[k])/len(vals[k])
        # print('{}: {}'.format(k, res))

    print(eventlog_name, '- total: {}'.format(sum(l2)/len(l2)))
