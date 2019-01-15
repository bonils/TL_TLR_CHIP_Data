#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:04:57 2018

@author: Steve
"""

# This script visualizes binding curves
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
#%% TAKES A LONG TIME SO ONLY RUN ONCE!!!!!!!!!


data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'

#binding data contains the averaged data for all clusters for each variant
#This data apparently still has not been averaged across datasets.
#So, for GAAA at 30 mM Mg, there are 2 datasets, and this suggests that the 
#data that I have been using for the rest of the analysis consists of the 
#average between two of the binding data files. 
binding_data = pd.read_csv(data_path + 'Mut2_GAAA_1.PerVariant.CPseries',
                           delim_whitespace = True)
binding_data_GUAA = pd.read_csv(data_path + 'Mut2_GUAA_1.PerVariant.CPseries',
                           delim_whitespace = True)

binding_data = binding_data.set_index('variant_number')
binding_data_GUAA = binding_data_GUAA.set_index('variant_number')

#%%
#the fitted data file contains the fitted parameters for each cluster. 
fitted_data_GAAA1 = pd.read_csv(data_path + 'Mut2_GAAA_1.CPfitted',
                           delim_whitespace = True) 

fitted_data_GAAA2 = pd.read_csv(data_path + 'Mut2_GAAA_2.CPfitted',
                           delim_whitespace = True) 
#%% Contains the parameters that went into the spreadsheets that I have been using
#for the rest of the analysis
error_scaled_GUAA = pd.read_csv(data_path + 'Mut2_GUAA_1.error_scaled.CPvariant',
                           delim_whitespace = True)

error_scaled_GAAA1 = pd.read_csv(data_path + 'Mut2_GAAA_1.error_scaled.CPvariant',
                           delim_whitespace = True)

error_scaled_GAAA2 = pd.read_csv(data_path + 'Mut2_GAAA_2.error_scaled.CPvariant',
                           delim_whitespace = True)




#%% get data from spreadsheet--> this is the summarized data
lib_data = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
lib_data['new_name'] = lib_data.r_seq + lib_data.r_name
print(lib_data.shape)
lib_data  = lib_data.drop_duplicates(subset='seq')
print(lib_data.shape)
lib_data = lib_data.set_index('variant_number')
WT_data = lib_data[lib_data.r_seq == 'UAUGG_CCUAAG']
#%%
all_data = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
all_data = all_data.drop_duplicates(subset='seq')
all_11ntR = pd.read_csv(data_path + 'all_11ntRs_unique.csv' )
mask = ((all_11ntR.b_name == 'normal') & (all_11ntR.no_mutations == 1)) | ((all_11ntR.b_name == 'normal') & (all_11ntR.no_mutations == 0))
single_11ntR_mutants = all_11ntR[mask].copy()
single_11ntR_mutants['new_name'] = single_11ntR_mutants['r_name'] + '_' + single_11ntR_mutants['r_seq']
unique_receptors = list(set(single_11ntR_mutants['r_seq']))

#%%get only the data that made it to the summary table after filtering
A = all_data.copy()
A = A.set_index('variant_number')
dG_30mM = A['dG_Mut2_GAAA']
dG_30mM = dG_30mM.dropna()

filtered_GAAA_1 = error_scaled_GAAA1.loc[dG_30mM.index]
filtered_GAAA_2 = error_scaled_GAAA2.loc[dG_30mM.index]
plt.scatter(filtered_GAAA_1['dG'],filtered_GAAA_2['dG'])
plt.xlim(-14,-6)
plt.ylim(-14,-6)
differences = filtered_GAAA_1['dG'] - filtered_GAAA_2['dG']
differences = differences.apply('abs') 

dG_30mM = A['dG_Mut2_GAAA'].copy()
dG_30mM[dG_30mM > 7.1] = np.nan
dG_30mM = dG_30mM.dropna()

filtered_GAAA_1 = error_scaled_GAAA1.loc[dG_30mM.index]
filtered_GAAA_2 = error_scaled_GAAA2.loc[dG_30mM.index]
differences = filtered_GAAA_1['dG'] - filtered_GAAA_2['dG']
differences = differences.apply('abs') 
fraction_below_0p5 = len(differences[differences > 0.5])/len(differences)
print('fraction below 0.5 kcal/mol: ' + str(1 - fraction_below_0p5))


fraction_below_1 = len(differences[differences > 1])/len(differences)
print('fraction below 1 kcal/mol: ' + str(1 - fraction_below_1))
#%%






#%%
#this table tracks each variant to all the cluster IDs correspondinf to that variant.
table_variants = pd.read_csv(data_path + 'tecto_undetermined.CPannot.CPannot',
                           delim_whitespace = True)
group_cluster_variants = table_variants.groupby(by='variant_number')
#%%
fitted_data_GAAA1 = fitted_data_GAAA1.set_index('clusterID')
fitted_data_GAAA2 = fitted_data_GAAA2.set_index('clusterID')

#%% I was doing this to see which data were missing
#after discussing with Sarah, we determined that the data that was missing
#belongs to variants that did not have enough clusters to pass the threshold
#of 5. 
nan_GUAA = all_data[all_data['dG_Mut2_GUAA_1'].isna()]
print('number of missing data with GUAA: ' + str(len(nan_GUAA)))
variants_nan_GUAA = list(nan_GUAA['variant_number'])
original_data_GUAA = pd.DataFrame(index = variants_nan_GUAA)
dG_list = []
ub_list = []
lb_list = []
cluster_list = []
for variants in variants_nan_GUAA:
    err_scaled_GUAA_variant = error_scaled_GUAA.loc[variants]['dG']
    up_bound = error_scaled_GUAA.loc[variants]['dG_ub']
    low_bound = error_scaled_GUAA.loc[variants]['dG_lb']
    dG_list.append(err_scaled_GUAA_variant)
    ub_list.append(up_bound)
    lb_list.append(low_bound)
    cluster_list.append(error_scaled_GUAA.loc[variants]['numClusters'])
original_data_GUAA['dG'] = dG_list
original_data_GUAA['dG_lb'] = lb_list
original_data_GUAA['dG_ub'] = ub_list
original_data_GUAA['numClusters'] = cluster_list

plt.hist(original_data_GUAA.numClusters.dropna(),bins=range(0,50))
#%%This code below I wrote to figure out that there was a correspondence between
#the values in the summary spreadsheet and the values in the error scaled files.
temp_WT = all_data[all_data.r_seq == 'UAUGG_CCUAAG']
temp_WT = temp_WT.set_index('variant_number')
idx = 2
print('test variant number :' + str(temp_WT.index[idx]))
dG_GUAA = temp_WT.loc[temp_WT.index[idx]]['dG_Mut2_GUAA_1']
print(dG_GUAA)

#the variant # agrees in both the cvs spreadsheet and the error scaled file
err_scaled_GUAA_variant = error_scaled_GUAA.loc[temp_WT.index[idx]]['dG']
print('entered in cvs table is :' + str(err_scaled_GUAA_variant))
#%%
#for variant # (240 in this case )
#find all clusters associated with that variant and plot the median dG
variant = 44500
clusters_variant = group_cluster_variants.get_group(variant)


data_variant = fitted_data_GAAA1.loc[list(clusters_variant['clusterID'])]
print('the median from individual clusters is ' + str(data_variant['dG'].median()))

data_variant2 = fitted_data_GAAA2.loc[list(clusters_variant['clusterID'])]
print('the median from individual clusters is ' + str(data_variant2['dG'].median()))

# this number should roughly agree with that in the spreadshet
dG_variant = all_data[all_data['variant_number'] ==  variant]['dG_Mut2_GAAA']
print('the value reported in the spreadsheet is ' + str(dG_variant))


#%%





#%%
temp = WT_data[WT_data.old_idx == '240']
#%%
#data point for WT at high affinity ~12 kcal/mol
WT_data = WT_data[WT_data.b_name == 'normal']
all_WT_variants = list(WT_data.index)
#%%
for variant in all_WT_variants[20:30]:
    fluorescence = binding_data.loc[variant]
    concentration = pd.Series([0.91e-9,2.74e-9,8.23e-9,24.7e-9,74.1e-9,222e-9,667e-9,2e-6])
    plt.figure()
    ax = plt.gca()
    ax.scatter(concentration,fluorescence)
    ax.set_xlim(0.1e-9,2000e-9)
    ax.set_xscale('log')
#%%
variant = 12563
fluorescence = binding_data.loc[variant]
concentration = pd.Series([0.91e-9,2.74e-9,8.23e-9,24.7e-9,74.1e-9,222e-9,667e-9,2e-6])
plt.figure()
ax = plt.gca()
ax.scatter(concentration,fluorescence)
ax.set_xlim(0.1e-9,2000e-9)
ax.set_xscale('log')
ax.set_title('GAAA') 
#%%   
    
fluorescence = binding_data_GUAA.loc[variant]
concentration = pd.Series([0.91e-9,2.74e-9,8.23e-9,24.7e-9,74.1e-9,222e-9,667e-9,2e-6])
plt.figure()
ax = plt.gca()
ax.scatter(concentration,fluorescence)
ax.set_xlim(0.1e-9,2000e-9)
ax.set_xscale('log')
#%%
receptor = 'CAUGG_CCUAAG'
receptor_data = single_11ntR_mutants[single_11ntR_mutants.r_seq == receptor].copy()
variants = list(receptor_data.variant_number)
dG_GUAA = list(receptor_data.dG_Mut2_GUAA_1)

counter = -1
for variant in variants:
    counter += 1
    fluorescence = binding_data_GUAA.loc[variant]
    concentration = pd.Series([0.91e-9,2.74e-9,8.23e-9,24.7e-9,74.1e-9,222e-9,667e-9,2e-6])
    plt.figure()
    ax = plt.gca()
    ax.scatter(concentration,fluorescence)
    ax.set_xlim(0.1e-9,2000e-9)
    ax.set_xscale('log')
    ax.set_title(str(dG_GUAA[counter]))
    
    
#%%
nan_GUAA = all_data[all_data['dG_Mut2_GUAA_1'].isna()]
print ('missing values with GUAA: ' + str(len(nan_GUAA)))
nan_GUAA_variants = list(nan_GUAA.variant_number)
nan_GUAA_dG = list(nan_GUAA.dG_Mut2_GUAA_1)
counter = -1
for variant in nan_GUAA_variants[10:20]:
    counter += 1
    fluorescence = binding_data_GUAA.loc[variant]
    concentration = pd.Series([0.91e-9,2.74e-9,8.23e-9,24.7e-9,74.1e-9,222e-9,667e-9,2e-6])
    plt.figure()
    ax = plt.gca()
    ax.scatter(concentration,fluorescence)
    ax.set_xlim(0.1e-9,2000e-9)
    ax.set_xscale('log')
    ax.set_title('variant number' + str (variant))
#%%
nan_GAAA = all_data[all_data['dG_Mut2_GAAA'].isna()]
print ('missing values with GAAA: ' + str(len(nan_GAAA)))
nan_GAAA_variants = list(nan_GAAA.variant_number)
nan_GAAA_dG = list(nan_GAAA.dG_Mut2_GAAA)
counter = -1
for variant in nan_GAAA_variants[20:30]:
    counter += 1
    fluorescence = binding_data.loc[variant]
    concentration = pd.Series([0.91e-9,2.74e-9,8.23e-9,24.7e-9,74.1e-9,222e-9,667e-9,2e-6])
    plt.figure()
    ax = plt.gca()
    ax.scatter(concentration,fluorescence)
    ax.set_xlim(0.1e-9,2000e-9)
    ax.set_xscale('log')
    ax.set_title('variant number: ' + str(variant))
#%%
    
x = all_data[all_data.variant_number == 11332]
