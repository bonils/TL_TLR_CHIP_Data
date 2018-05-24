#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 13:07:41 2018

@author: Steve
"""

#%%
'''--------------Import Libraries--------------------'''
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

#%%
'''--------------Import Functions--------------''' 
from clustering_functions import get_dG_for_scaffold
from clustering_functions import doPCA
from clustering_functions import interpolate_mat_knn
from clustering_functions import prep_data_for_clustering_ver2

#%%
'''---------------General Variables-------------'''
dG_threshold = -7.1 #kcal/mol; dG values above this are not reliable
dG_replace = -7.1 # for replacing values above threshold. 
nan_threshold = 0.50 #amount of missing data tolerated.

#%%
'''---------------Import data ------------------'''
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
tecto_data = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
sublib0 = tecto_data[tecto_data['sublibrary'] == 'tertcontacts_0']
print('original size of sublibrary 0: ',sublib0.shape)
sublib0 = sublib0.drop_duplicates(subset='seq', keep="last")
print('after deleting duplicates: ',sublib0.shape)
#%%
receptors_sublib0 = sorted(list(set(sublib0['r_name'])))
scaffolds_sublib0 = sorted(list(set(sublib0['old_idx'])))
#%%
#Note: in previous datatables (e.g. all_11ntR_unique.csv) old_idx imports as a 
# number.  However in the case of the original table 'tectorna_results_tertcontacts.180122.csv'
# old_idx imports as a string.
data = sublib0
conditions = ['dG_Mut2_GAAA', 'dG_Mut2_GUAA_1']
column_labels = ['dG_30mM_Mg_GAAA','dG_30mM_Mg_GUAA']
row_index= ('r_name')
flanking = 'normal'

data_50_scaffolds = pd.DataFrame(index = receptors_sublib0)

for scaffolds in scaffolds_sublib0:
    
    next_scaffold = str(scaffolds)
    next_column_labels = [name + '_' +  next_scaffold for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(receptors_sublib0)
    data_50_scaffolds = pd.concat([data_50_scaffolds,next_df],axis = 1)
#%%
prep_data,original_nan = prep_data_for_clustering_ver2(data_50_scaffolds,
                                                       dG_threshold,dG_replace,nan_threshold)

#%%

#(3) Subtract the mean per row
norm_data = prep_data.sub(prep_data.mean(axis = 1),axis = 0)
norm_data_nan = norm_data.copy()
norm_data_nan[original_nan] = np.NAN
#%%
sublib0.to_csv('sublib0.csv')





