'''
this script takes as input the output of evaluate_suffix_and_remaining_time.py
therefore, the latter needs to be executed first

Author: Niek Tax
'''

from __future__ import division

import csv
import os


def calc_dl(output_file):
    csvfile = open(os.path.join(output_file, "suffix_predictions.csv"), 'r')
    r = csv.reader(csvfile)
    next(r,None) # header
    vals = []
    for row in r:
        vals.append(float(row[5]))

    return sum(vals)/len(vals)
