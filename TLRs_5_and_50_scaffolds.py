#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:11:51 2018

@author: Steve
"""

#%%
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
#%%
'''--------------Import Functions--------------''' 
from clustering_functions import get_dG_for_scaffold
from clustering_functions import doPCA
from clustering_functions import interpolate_mat_knn
from clustering_functions import prep_data_for_clustering_ver2
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
#%%Drop repeats, just because they are there; is the same data
entire_lib = entire_lib.drop_duplicates(subset ='seq')
#%%Consider the ones only with normal closing base pair 
mask = entire_lib.b_name == 'normal' 
entire_lib_normal_bp = entire_lib[mask]
#%%Exclude sublibrary 5 which has the mutation intermediates and cannot be easily classified (or maybe they can be classified as others)
mask = (entire_lib_normal_bp.sublibrary == 'tertcontacts_0') | (entire_lib_normal_bp.sublibrary == 'tertcontacts_1')|\
       (entire_lib_normal_bp.sublibrary == 'tertcontacts_2') | (entire_lib_normal_bp.sublibrary == 'tertcontacts_3')|\
       (entire_lib_normal_bp.sublibrary == 'tertcontacts_4') | (entire_lib_normal_bp.sublibrary == 'tertcontacts_5')
entire_lib_selected = entire_lib_normal_bp[mask].copy()
#%%Create new name identifier because there are some that were given the same name
#but they are different receptors
#also attacht the sublibrary they came from for later reference
entire_lib_selected['new_name'] = entire_lib_selected.r_name + '_' + entire_lib_selected.r_seq + '_' + entire_lib_selected.sublibrary
#%%
unique_receptors = sorted(list(set(list(entire_lib_selected.new_name))))
#%% Make a list with all scaffolds with scaffolds 1 to 5 defined
scaffolds_five = ['13854','14007','14073','35311_A','35600']
all_scaffolds = list(set(entire_lib_selected.old_idx))
all_scaffolds.remove('13854')
all_scaffolds.remove('14007')
all_scaffolds.remove('14073')
all_scaffolds.remove('35311_A')
all_scaffolds.remove('35600')
print(len(all_scaffolds))
all_scaffolds = ['13854', '14007','14073','35311_A','35600'] + all_scaffolds
print(len(all_scaffolds))
#%% Create dataframe with all scaffolds, grouped by TLR
data = entire_lib_selected
conditions = ['dG_Mut2_GAAA']#, 'dG_Mut2_GUAA_1']
column_labels = ['dG_30mM_Mg_GAAA']#,'dG_30mM_Mg_GUAA']
row_index= ('new_name')
flanking = 'normal'

data_50_scaffolds_GAAA = pd.DataFrame(index = unique_receptors)

for scaffolds in all_scaffolds:
    
    next_scaffold = str(scaffolds)
    next_column_labels = [name + '_' +  next_scaffold for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_GAAA = pd.concat([data_50_scaffolds_GAAA,next_df],axis = 1)

conditions = ['dG_Mut2_GUAA_1']
column_labels = ['dG_30mM_Mg_GUAA']
row_index= ('new_name')
flanking = 'normal'

data_50_scaffolds_GUAA = pd.DataFrame(index = unique_receptors)

for scaffolds in all_scaffolds:
    
    next_scaffold = str(scaffolds)
    next_column_labels = [name + '_' +  next_scaffold for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_GUAA = pd.concat([data_50_scaffolds_GUAA,next_df],axis = 1)


conditions = ['dG_Mut2_GAAA_5mM_2']
column_labels = ['dG_5mM_Mg_GAAA']
row_index= ('new_name')
flanking = 'normal'

data_50_scaffolds_5Mg = pd.DataFrame(index = unique_receptors)

for scaffolds in all_scaffolds:
    
    next_scaffold = str(scaffolds)
    next_column_labels = [name + '_' +  next_scaffold for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_5Mg = pd.concat([data_50_scaffolds_5Mg,next_df],axis = 1)


conditions = ['dG_Mut2_GAAA_5mM_150mMK_1']
column_labels = ['dG_5Mg150K_GAAA']
row_index= ('new_name')
flanking = 'normal'

data_50_scaffolds_5Mg150K = pd.DataFrame(index = unique_receptors)

for scaffolds in all_scaffolds:
    
    next_scaffold = str(scaffolds)
    next_column_labels = [name + '_' +  next_scaffold for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_5Mg150K = pd.concat([data_50_scaffolds_5Mg150K,next_df],axis = 1)
    
data_50_scaffolds_allTLRs = pd.concat([data_50_scaffolds_GAAA,
                               data_50_scaffolds_5Mg,
                               data_50_scaffolds_5Mg150K,
                               data_50_scaffolds_GUAA],axis = 1)
#%%
prep_data_50_scaff_all_TLRs,original_nan_50_scaff_all_TLRs = prep_data_for_clustering_ver2(data_50_scaffolds_allTLRs,
                                                       dG_threshold,dG_replace,nan_threshold,
                                                       num_neighbors=num_neighbors)
#%% CATALOGUING RECEPTORS: Create a receptor types dataframe with the type.
all_receptors = list(prep_data_50_scaff_all_TLRs.index)
receptor_types_df = pd.DataFrame(index = prep_data_50_scaff_all_TLRs.index)
receptor_types_df['type'] = 'uncatalogued'

for next_receptor in receptor_types_df.index:
    if 'tertcontacts_5' in next_receptor:
        receptor_types_df.loc[next_receptor]['type'] = 'other'
    else:
        if '11ntR' in next_receptor:
            receptor_types_df.loc[next_receptor]['type'] = '11ntR' 
            
        if 'IC3' in next_receptor:
            receptor_types_df.loc[next_receptor]['type'] = 'IC3' 
    
        if 'VC2' in next_receptor:
            receptor_types_df.loc[next_receptor]['type'] = 'VC2'
    
        if ('C7.' in next_receptor) | ('B7.' in next_receptor)| ('GAAC_receptor' in next_receptor) :
            receptor_types_df.loc[next_receptor]['type'] = 'in_vitro' 
receptor_counts = receptor_types_df['type'].value_counts()
print(receptor_counts)
receptor_counts = receptor_counts.drop(['uncatalogued'])
color_list = ['blue','red','yellow','black','green']
plt.pie(receptor_counts,colors=color_list,startangle = 40,wedgeprops={'linewidth': 1,'edgecolor':'black', 'linestyle': 'solid'})
plt.axis('equal')
    #%% Drop 'uncatalogued' receptors from dataframe to be plotted and analyzed
# These uncatalogued receptors contain tandem bp receptors, and random crap that I got
# from the sequence database of group I introns
uncatalogued_receptors = receptor_types_df[receptor_types_df.type == 'uncatalogued']
prep_data_50_scaff_all_TLRs = prep_data_50_scaff_all_TLRs.drop(uncatalogued_receptors.index)
#%% Also drop uncatalogued receptors from the receptor_types dataframe
receptor_types_df = receptor_types_df.drop(uncatalogued_receptors.index)
#%% reinsert nan values VERY IMPORTANT!!!!!
# Heatmaps and many of the other plots need not to included nan values
# nans were assigned values by interpolation only for clustering and other 
# statistical tools that do not support missing values.
prep_data_all_with_nan = prep_data_50_scaff_all_TLRs.copy()
prep_data_all_with_nan[original_nan_50_scaff_all_TLRs] = np.nan
#%%
sublibrary = []
for receptors in prep_data_all_with_nan.index:
    if 'tertcontacts_0' in receptors:
        sublibrary.append(0)
    if 'tertcontacts_1' in receptors:
        sublibrary.append(1)
    if 'tertcontacts_2' in receptors:
        sublibrary.append(2)
    if 'tertcontacts_3' in receptors:
        sublibrary.append(3)
    if 'tertcontacts_4' in receptors:
        sublibrary.append(4)
    if 'tertcontacts_5' in receptors:
        sublibrary.append(5)  
#%%        
prep_data_all_with_nan['sublibrary'] = sublibrary
#%% Calculate r pearson between 5mM and 5mM + potassium
# exclude values that are limits
correlation_coeffs 







