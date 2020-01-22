'''
this script trains an LSTM model on one of the data files in the data folder of
this repository. the input file can be changed to another file from the data folder
by changing its name in line 46.

it is recommended to run this script on GPU, as recurrent networks are quite 
computationally intensive.

Author: Niek Tax
'''

from __future__ import print_function, division
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input
from keras.utils.data_utils import get_file
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from collections import Counter
import unicodecsv
import numpy as np
import random
import sys
import os
import copy
import csv
import time
from datetime import datetime
from math import log


train_eventlog = "../../Camargo/output_files/HELPDESK_20000000_0_0_0_0_1/shared_cat/data/train_log.csv"
test_eventlog = "../../Camargo/output_files/HELPDESK_20000000_0_0_0_0_1/shared_cat/data/test_log.csv"
caseid_col = 1
role_col = 3
task_col = 4
########################################################################################
#
# this part of the code opens the file, reads it into three following variables
#

lines = [] #these are all the activity seq

#helper variables
lastcase = ''
line = ''
firstLine = True
numlines = 0

csvfile = open(train_eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
ascii_offset = 161

for row in spamreader: #the rows are "CaseID,ActivityID,CompleteTimestamp"
    if row[caseid_col]!=lastcase:  #'lastcase' is to save the last executed case for the loop
        lastcase = row[caseid_col]
        if not firstLine:
            lines.append(line)
        line = ''
        numlines+=1
    line+=chr(int(row[task_col])+ascii_offset)
    firstLine = False

# add last case
lines.append(line)
numlines+=1


#########################################################################################################

# separate training data into 3 parts

step = 1
sentences = []
softness = 0
next_chars = []
lines = list(map(lambda x: x+'!',lines)) #put delimiter symbol
maxlen = max(map(lambda x: len(x),lines)) #find maximum line size

# next lines here to get all possible characters for events and annotate them with numbers
chars = map(lambda x: set(x),lines)
chars = list(set().union(*chars))
chars.sort()
target_chars = copy.copy(chars)
chars.remove('!')
print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
print(indices_char)


csvfile = open(train_eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
lastcase = ''
line = ''
firstLine = True
lines = []
numlines = 0
for row in spamreader:
    if row[caseid_col]!=lastcase:
        lastcase = row[caseid_col]
        if not firstLine:
            lines.append(line)
        line = ''
        numlines+=1
    line+=chr(int(row[task_col])+ascii_offset)
    firstLine = False

# add last case
lines.append(line)
numlines+=1

step = 1
sentences = []
softness = 0
next_chars = []
lines = map(lambda x: x+'!',lines)

for line in lines:
    for i in range(0, len(line), step):
        if i==0:
            continue

        #we add iteratively, first symbol of the line, then two first, three...
        sentences.append(line[0: i])
        next_chars.append(line[i])
print('nb sequences:', len(sentences))

print('Vectorization...')
num_features = len(chars)+1
print('num features: {}'.format(num_features))
X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
y_t = np.zeros((len(sentences)), dtype=np.float32)
for i, sentence in enumerate(sentences):
    leftpad = maxlen-len(sentence)
    for t, char in enumerate(sentence):
        multiset_abstraction = Counter(sentence[:t+1])
        for c in chars:
            if c==char: #this will encode present events to the right places
                X[i, t+leftpad, char_indices[c]] = 1
                break
        X[i, t+leftpad, len(chars)] = t+1
    for c in target_chars:
        if c==next_chars[i]:
            y_a[i, target_char_indices[c]] = 1-softness
        else:
            y_a[i, target_char_indices[c]] = softness/(len(target_chars)-1)
    #np.set_printoptions(threshold=np.nan)

# build the model: 
print('Build model...')
main_input = Input(shape=(maxlen, num_features), name='main_input')
# train a 2-layer LSTM with one shared layer
l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input) # the shared layer
b1 = BatchNormalization()(l1)
l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in activity prediction
b2_1 = BatchNormalization()(l2_1)
#l2_2 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in time prediction
#b2_2 = BatchNormalization()(l2_2)
act_output = Dense(len(target_chars), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)
#time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)

model = Model(inputs=[main_input], outputs=[act_output])

opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

model.compile(loss={'act_output':'categorical_crossentropy'}, optimizer=opt)
early_stopping = EarlyStopping(monitor='val_loss', patience=42)
model_checkpoint = ModelCheckpoint('output_files/models/model_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

model.fit(X, {'act_output':y_a}, validation_split=0.2, verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=maxlen, epochs=500)