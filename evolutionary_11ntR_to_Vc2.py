#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:43:31 2018

@author: Steve
"""
# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import *
import scipy.cluster.hierarchy as sch
import sklearn.decomposition as skd
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools
#%%
'''--------------Import Functions--------------''' 
from clustering_functions import get_dG_for_scaffold
from clustering_functions import doPCA
from clustering_functions import interpolate_mat_knn
from clustering_functions import prep_data_for_clustering_ver2
from clustering_functions import prep_data_for_clustering_ver3
#%%
#General Variables
dG_threshold = -7.1 #kcal/mol; dG values above this are not reliable
dG_replace = -7.1 # for replacing values above threshold. 
nan_threshold = 1 #amount of missing data tolerated.
num_neighbors = 10 # for interpolation
#%%
#import data from csv files 
#Data has been separated into 11ntR, IC3, and in vitro
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
entire_lib = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
entire_lib = entire_lib[entire_lib['b_name'] == 'normal']
#%% Separate sublibrary 5, which contains sequence interemediates
sublib5 = entire_lib[entire_lib['sublibrary'] == 'tertcontacts_5' ]
receptors = set(sublib5.r_name)
Vc2_to_11ntR = []
# Look for intermediates between 11ntR and Vc2
for i in receptors:
    if ('Vc2' in i) and ('11nt_R' in i):
        Vc2_to_11ntR.append(i)
Vc2_to_11ntR_data = sublib5[sublib5.r_name.isin(Vc2_to_11ntR)]
#%%
#additional receptors to add; these are intermediates that are not in 
#sublib 5 but somewhere else in the library
additional = ['UAUGG_CCUAAC', 'GAUGG_CCUAAG', 'UAUGG_CCUAAG', 'GAUGG_CCUAAC']
B = entire_lib.groupby('r_seq')
for receptor in additional:
    more_data = B.get_group(receptor)
    Vc2_to_11ntR_data = pd.concat([Vc2_to_11ntR_data,more_data])
#%% Make a list of the sequences of the intermediates.
intermediate_seq = set(Vc2_to_11ntR_data.r_seq)
intermediate_group = Vc2_to_11ntR_data.groupby('r_name')

#Generarate sequences to put into mFOLD for secondary structure predictions.
hairpins = []
for sequence in intermediate_seq:
    left = sequence[0:5]
    right = sequence[6:]
    new_str ='cgcgc'+ left + 'ccguucgcgg' + right + 'gcgcg;'
    hairpins.append(new_str)
groups = Vc2_to_11ntR_data.groupby('r_seq')
#%% Calculate average data for the 32 intermediates between 11ntR and Vc2
average_data = pd.DataFrame(index = intermediate_seq)

dG_avg_30Mg = []
dG_avg_5Mg = []
dG_avg_5Mg150K = []
dG_avg_GUAA = []

for receptor in average_data.index:
    receptor_data = groups.get_group(receptor)
    dG_avg_30Mg.append(receptor_data['dG_Mut2_GAAA'].mean())

average_data['dG_avg_30Mg'] = dG_avg_30Mg
#%% Functions to perform to 11ntR to arrive to Vc2
#step A
def insert_A (seq):
    new_seq = seq[0:10] + 'A' + seq[10:]
    return new_seq
#step B
def mutG6C (seq):
    new_seq = seq[0:-1] + 'C'
    return new_seq
#step C
def mutU7G (seq):
    new_seq = 'G' + seq[1:]
    return new_seq
#step D
def mutA8U (seq):
    new_seq = seq[0] + 'U' + seq[2:]
    return new_seq
#step E
def mutU9A (seq):
    new_seq = seq[0:2] + 'A' + seq[3:]
    return new_seq

#%% Very first path to investigate 
seq = 'UAUGG_CCUAAG'
seq_path_1 = [seq]

seq1 = insert_A(seq)
seq_path_1.append(seq1)
seq2 = mutG6C(seq1)
seq_path_1.append(seq2)
seq3 = mutU7G(seq2)
seq_path_1.append(seq3)
seq4 = mutA8U(seq3)
seq_path_1.append(seq4)
seq5 = mutU9A(seq4)
seq_path_1.append(seq5)   
print(seq_path_1)
#%% Create permutations for all possible paths
path = ['A','B','C','D','E']
list_paths = list(itertools.permutations(path))
first_seq = 'UAUGG_CCUAAG'
list_paths_seq = []

for path in list_paths:
    list_intermediates = [first_seq]
    seq = first_seq
    for step in path:
        if step == 'A':
            seq = insert_A(seq)
        elif step == 'B':
            seq = mutG6C(seq)
        elif step == 'C':
            seq = mutU7G(seq)
        elif step == 'D':
            seq = mutA8U(seq)
        elif step == 'E':
            seq = mutU9A(seq)
        list_intermediates.append(seq)
    list_paths_seq.append(list_intermediates)            
#%%
flat_list = [item for sublist in list_paths_seq for item in sublist]   
A = set(flat_list)
not_found = []
for items in A:
    if items in intermediate_seq:
        pass
    else:
        not_found.append(items)
 
#%%
dG_path_list = []        
for path in list_paths_seq:
    dG_across_path = []
    for sequence in path:
        dG_across_path.append(average_data.loc[sequence]['dG_avg_30Mg'])
    dG_path_list.append(dG_across_path)

#%%
plt.figure()
for dGs in dG_path_list:
    plt.plot(range(6),dGs,'-o')
plt.show()

#%%
#FIND THE LOWEST PATH
    
        



