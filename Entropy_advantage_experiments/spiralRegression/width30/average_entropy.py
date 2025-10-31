# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:50:47 2023

@author: entao.yang
"""

import numpy as np
import os
import pickle
import sys

if (len(sys.argv) != 2):
    print('Usage: average_entropy.py process_num')
    sys.exit()

process_num = int(sys.argv[1])

# assume we are parallellized on 2 processes here
for i in range(process_num):
    if i == 0:
        with open('./process_%d/entropy.pickle' %i, 'rb') as pickled_file:
             entropy = pickle.load(pickled_file)
    else:
        with open('./process_%d/entropy.pickle' %i, 'rb') as pickled_file:
             entropy_buffer = pickle.load(pickled_file)
        entropy += entropy_buffer

entropy = entropy / process_num

with open('./ave_entropy.pickle', 'wb') as pickle_file:
    pickle.dump(entropy,  pickle_file)
