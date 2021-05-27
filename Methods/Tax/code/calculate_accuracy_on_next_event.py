'''
this script takes as input the output of evaluate_suffix_and_remaining_time.py
therefore, the latter needs to be executed first

Author: Niek Tax
'''

from __future__ import division

import csv
import os


def calc_accuracy(output_file):
    csvfile = open(os.path.join(output_file, "predictions.csv"), 'r')
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

    l2 = list()
    for k in vals.keys():
        l2.extend(vals[k])

    return sum(l2)/len(l2)
