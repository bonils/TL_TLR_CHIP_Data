#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:08:28 2018

@author: Steve
"""

'''--------------Import Libraries--------------------'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import random 
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
num_neighbors = 10
#for plotting
low_lim = -14
high_lim = -6
#%%
'''---------------Import data ------------------'''
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
all_data = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
all_11ntR = pd.read_csv(data_path + 'all_11ntRs_unique.csv' )
mask = ((all_11ntR.b_name == 'normal') & (all_11ntR.no_mutations == 1)) | ((all_11ntR.b_name == 'normal') & (all_11ntR.no_mutations == 0))
single_11ntR_mutants = all_11ntR[mask].copy()
single_11ntR_mutants['new_name'] = single_11ntR_mutants['r_name'] + '_' + single_11ntR_mutants['r_seq']
unique_receptors = list(set(single_11ntR_mutants['r_seq']))

scaffolds_five = ['13854','14007','14073','35311_A','35600']
all_scaffolds = list(set(single_11ntR_mutants.old_idx))
all_scaffolds.remove(13854)
all_scaffolds.remove(14007)
all_scaffolds.remove(14073)
all_scaffolds.remove(35311)
all_scaffolds.remove(35600)
print(len(all_scaffolds))
all_scaffolds = [13854, 14007,14073,35311,35600] + all_scaffolds
#%%
#export 11ntR sequences for frequency analysis with matlab
seq_11ntR = {'sequences':list(set(all_11ntR.r_seq))}
sequences_11ntR_df =pd.DataFrame(seq_11ntR)
sequences_11ntR_df.to_csv('all_11ntR_sequences.csv')
#%% Select controls, tandem base pairs, for comparing scaffolds preferences
print(len(all_data))
all_data = all_data.drop_duplicates(subset = 'seq')
print(len(all_data))
#Get tandem base pair stuff
sublib_0 = all_data[all_data['sublibrary'] == 'tertcontacts_0'].copy()
receptors_sublib0 = list(set(sublib_0['r_name']))

tandem_receptors = []
for receptor in receptors_sublib0:
    if ('tandem' in receptor) | ('T4' in receptor):
        tandem_receptors.append(receptor)

sublib0_grouped = sublib_0.groupby('r_name')
#%% Get all data in single dataframe
data = single_11ntR_mutants
conditions = ['dG_Mut2_GAAA']
column_labels = ['dG_30mM_Mg_GAAA']
row_index= ('r_seq')
flanking = 'normal'

data_50_scaffolds_GAAA = pd.DataFrame(index = unique_receptors)

for scaffolds in all_scaffolds:
    
    next_scaffold = scaffolds
    next_column_labels = [name + '_' +  str(next_scaffold) for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_GAAA = pd.concat([data_50_scaffolds_GAAA,next_df],axis = 1)

conditions = ['dG_Mut2_GUAA_1']
column_labels = ['dG_30mM_Mg_GUAA']
row_index= ('r_seq')
flanking = 'normal'

data_50_scaffolds_GUAA = pd.DataFrame(index = unique_receptors)

for scaffolds in all_scaffolds:
    
    next_scaffold = scaffolds
    next_column_labels = [name + '_' +  str(next_scaffold) for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_GUAA = pd.concat([data_50_scaffolds_GUAA,next_df],axis = 1)


conditions = ['dG_Mut2_GAAA_5mM_2']
column_labels = ['dG_5mM_Mg_GAAA']
row_index= ('r_seq')
flanking = 'normal'

data_50_scaffolds_5Mg = pd.DataFrame(index = unique_receptors)

for scaffolds in all_scaffolds:
    
    next_scaffold = scaffolds
    next_column_labels = [name + '_' +  str(next_scaffold) for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_5Mg = pd.concat([data_50_scaffolds_5Mg,next_df],axis = 1)


conditions = ['dG_Mut2_GAAA_5mM_150mMK_1']
column_labels = ['dG_5Mg150K_GAAA']
row_index= ('r_seq')
flanking = 'normal'

data_50_scaffolds_5Mg150K = pd.DataFrame(index = unique_receptors)

for scaffolds in all_scaffolds:
    
    next_scaffold = scaffolds
    next_column_labels = [name + '_' +  str(next_scaffold) for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_5Mg150K = pd.concat([data_50_scaffolds_5Mg150K,next_df],axis = 1)
    
data_sM_11ntR = pd.concat([data_50_scaffolds_GAAA,
                               data_50_scaffolds_5Mg,
                               data_50_scaffolds_5Mg150K,
                               data_50_scaffolds_GUAA],axis = 1)

#%% Get all Errors in single dataframe
data = single_11ntR_mutants
conditions = ['dGerr_Mut2_GAAA']
column_labels = ['dG_30mM_Mg_GAAA']
row_index= ('r_seq')
flanking = 'normal'

data_50_scaffolds_GAAA = pd.DataFrame(index = unique_receptors)

for scaffolds in all_scaffolds:
    
    next_scaffold = scaffolds
    next_column_labels = [name + '_' +  str(next_scaffold) for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_GAAA = pd.concat([data_50_scaffolds_GAAA,next_df],axis = 1)

conditions = ['dGerr_Mut2_GUAA_1']
column_labels = ['dG_30mM_Mg_GUAA']
row_index= ('r_seq')
flanking = 'normal'

data_50_scaffolds_GUAA = pd.DataFrame(index = unique_receptors)

for scaffolds in all_scaffolds:
    
    next_scaffold = scaffolds
    next_column_labels = [name + '_' +  str(next_scaffold) for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_GUAA = pd.concat([data_50_scaffolds_GUAA,next_df],axis = 1)


conditions = ['dGerr_Mut2_GAAA_5mM_2']
column_labels = ['dG_5mM_Mg_GAAA']
row_index= ('r_seq')
flanking = 'normal'

data_50_scaffolds_5Mg = pd.DataFrame(index = unique_receptors)

for scaffolds in all_scaffolds:
    
    next_scaffold = scaffolds
    next_column_labels = [name + '_' +  str(next_scaffold) for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_5Mg = pd.concat([data_50_scaffolds_5Mg,next_df],axis = 1)


conditions = ['dGerr_Mut2_GAAA_5mM_150mMK_1']
column_labels = ['dG_5Mg150K_GAAA']
row_index= ('r_seq')
flanking = 'normal'

data_50_scaffolds_5Mg150K = pd.DataFrame(index = unique_receptors)

for scaffolds in all_scaffolds:
    
    next_scaffold = scaffolds
    next_column_labels = [name + '_' +  str(next_scaffold) for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_5Mg150K = pd.concat([data_50_scaffolds_5Mg150K,next_df],axis = 1)
    
error_sM_11ntR = pd.concat([data_50_scaffolds_GAAA,
                               data_50_scaffolds_5Mg,
                               data_50_scaffolds_5Mg150K,
                               data_50_scaffolds_GUAA],axis = 1)
#%%
WT_data = data_sM_11ntR.loc['UAUGG_CCUAAG']
WT_original_data = single_11ntR_mutants[single_11ntR_mutants.r_seq == 'UAUGG_CCUAAG']
WT_original_data = WT_original_data.set_index('old_idx')
WT_original_data = WT_original_data.reindex(all_scaffolds)

WT_error = error_sM_11ntR.loc['UAUGG_CCUAAG']
WT_original_data = single_11ntR_mutants[single_11ntR_mutants.r_seq == 'UAUGG_CCUAAG']
WT_original_data = WT_original_data.set_index('old_idx')
WT_original_data = WT_original_data.reindex(all_scaffolds)
#%%
#Color based on the length of the CHIP piece 
Colors = WT_original_data.length.copy()
Lengths = WT_original_data.length.copy()
Colors[Colors == 8] = 'red'
Colors[Colors == 9] = 'blue'
Colors[Colors == 10] = 'orange'
Colors[Colors == 11] = 'black'
#%%
#reorder single mutants to follow figure in paper
new_order = ['AAUGG_CCUAAG','CAUGG_CCUAAG','GAUGG_CCUAAG',
             'UCUGG_CCUAAG','UGUGG_CCUAAG','UUUGG_CCUAAG',
             'UAAGG_CCUAAG','UACGG_CCUAAG','UAGGG_CCUAAG',
             'UAUAG_CCUAAG','UAUCG_CCUAAG','UAUUG_CCUAAG',
             'UAUGA_CCUAAG','UAUGC_CCUAAG','UAUGU_CCUAAG',
             'UAUGG_ACUAAG','UAUGG_GCUAAG','UAUGG_UCUAAG',
             'UAUGG_CAUAAG','UAUGG_CGUAAG','UAUGG_CUUAAG',
             'UAUGG_CCAAAG','UAUGG_CCCAAG','UAUGG_CCGAAG',
             'UAUGG_CCUCAG','UAUGG_CCUGAG','UAUGG_CCUUAG',
             'UAUGG_CCUACG','UAUGG_CCUAGG','UAUGG_CCUAUG',
             'UAUGG_CCUAAA','UAUGG_CCUAAC','UAUGG_CCUAAU',
             'UAUGG_CCUAAG']
#%%
data_sM_11ntR = data_sM_11ntR.reindex(new_order)
error_sM_11ntR = error_sM_11ntR.reindex(new_order)
#%%
low_lim = -14
high_lim = -5
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
columns_to_plot = data_sM_11ntR.columns#[0:50]
#%% Plot only GAAA Tetraloops
GAAA_columns = []
for column in data_sM_11ntR.columns:
    if 'GUAA' not in column:
        GAAA_columns = GAAA_columns + [column] 
columns_to_plot = GAAA_columns        
#%%if you only want to plot 30 mM and 5mM 
columns_to_plot = GAAA_columns[0:100]
#%%
#WT_data[WT_data>-7.1] = np.nan
#data_sM_11ntR[data_sM_11ntR>-7.1] = np.nan
#%%
#%%if you only want to plot 30 mM 
columns_to_plot = GAAA_columns[0:50]
#%% Plot without error bars
ddG_median_list = []
ddG_std_list = []
n_compared_list = []
x_subplots = 3
y_subplots = 3
fig, axs = plt.subplots(x_subplots, y_subplots, sharey=True, sharex = True, tight_layout=True)
axs = axs.ravel()
counter = -1
figure_counter = 0
for receptors in data_sM_11ntR.index:
    counter += 1
    if counter < (x_subplots * y_subplots):
        #plot 30 mM GAAA
        axs[counter].scatter(WT_data[columns_to_plot[0:50]],data_sM_11ntR.loc[receptors][columns_to_plot[0:50]],s=120,edgecolors='k',c=list(Colors.values))
        #plot 5 mM GAAA
        axs[counter].scatter(WT_data[columns_to_plot[50:100]],data_sM_11ntR.loc[receptors][columns_to_plot[50:100]],s=120,edgecolors='k',c=list(Colors.values),marker='s')
        #plot 5 mM + 150K
        axs[counter].scatter(WT_data[columns_to_plot[100:150]],data_sM_11ntR.loc[receptors][columns_to_plot[100:150]],s=120,edgecolors='k',c=list(Colors.values),marker ='*')
        axs[counter].plot(x,x,':k')
        axs[counter].plot(x,y_thres,':k',linewidth = 0.5)
        axs[counter].plot(y_thres,x,':k',linewidth = 0.5)
        axs[counter].set_xlim(low_lim,high_lim)
        axs[counter].set_ylim(low_lim,high_lim)
        axs[counter].set_xticks(list(range(-14,-4,4)))
        axs[counter].set_yticks(list(range(-14,-4,4)))
        axs[counter].set_title(receptors)
        print(str(receptors))
        x_data = WT_data[columns_to_plot].copy()
        y_data = data_sM_11ntR.loc[receptors][columns_to_plot].copy()
        with_data = ~((x_data.isna()) | (y_data.isna()))
        data_points_total = with_data.sum() 
        ddG = y_data.subtract(x_data)
        
        #Calculate mediad ddG with all data
        ddG_avg = ddG.median()
        ddG_median_list.append(ddG_avg)
        ddG_std = ddG.std()
        ddG_std_list.append(ddG_std)
        
        #Get rid of data below limit to calculate correlations
        x_data[x_data > dG_threshold] = np.nan
        y_data[y_data > dG_threshold] = np.nan
        
        #Calculate ddGs with data below limit deleted
        ddG_with_thr = y_data.subtract(x_data)        
        above_limit = ~ddG_with_thr.isnull()
        n_above_limit = above_limit.sum()
        n_compared_list.append(n_above_limit)
        
        r_pearson = x_data.corr(y_data)
        r_sq = r_pearson**2
        textstr = 'n=%.0f\nn$_{thr}$ =%.0f\n$\Delta\Delta G=%.2f$\n$\mathrm{r}=%.2f$' % (data_points_total,n_above_limit,ddG_avg, r_sq)
        axs[counter].text(-9, -10, textstr, fontsize=7,
        verticalalignment='top')
        x_ddG = [low_lim + ddG_avg, dG_threshold]
        x_new = [low_lim, dG_threshold - ddG_avg]
        axs[counter].plot(x_new,x_ddG,'--r',linewidth = 3)

        
    else:
        figure_counter += 1
#        fig.savefig('/Users/Steve/Desktop/Tecto_temp_figures/single_mutant_profiles_3_' + str(figure_counter) + '.svg')
        fig, axs = plt.subplots(x_subplots, y_subplots, sharey=True, sharex = True, tight_layout=True)
        axs = axs.ravel()
                #plot 30 mM GAAA
        axs[0].scatter(WT_data[columns_to_plot[0:50]],data_sM_11ntR.loc[receptors][columns_to_plot[0:50]],s=120,edgecolors='k',c=list(Colors.values))
        #plot 5 mM GAAA
        axs[0].scatter(WT_data[columns_to_plot[50:100]],data_sM_11ntR.loc[receptors][columns_to_plot[50:100]],s=120,edgecolors='k',c=list(Colors.values),marker='s')
        #plot 5 mM + 150K
        axs[0].scatter(WT_data[columns_to_plot[100:150]],data_sM_11ntR.loc[receptors][columns_to_plot[100:150]],s=120,edgecolors='k',c=list(Colors.values),marker ='*')
        axs[0].plot(x,x,':k')
        axs[0].plot(x,y_thres,':k',linewidth = 0.5)
        axs[0].plot(y_thres,x,':k',linewidth = 0.5)
        axs[0].set_xlim(low_lim,high_lim)
        axs[0].set_ylim(low_lim,high_lim)
        axs[0].set_xticks(list(range(-14,-4,4)))
        axs[0].set_yticks(list(range(-14,-4,4)))
        axs[0].set_title(receptors)
        print(str(receptors))
        x_data = WT_data[columns_to_plot].copy()
        y_data = data_sM_11ntR.loc[receptors][columns_to_plot].copy()
        with_data = ~((x_data.isna()) | (y_data.isna()))
        data_points_total = with_data.sum() 
        ddG = y_data.subtract(x_data)
        
        #Calculate mediad ddG with all data
        ddG_avg = ddG.median()
        ddG_median_list.append(ddG_avg)
        ddG_std = ddG.std()
        ddG_std_list.append(ddG_std)
        
        #Get rid of data below limit to calculate correlations
        x_data[x_data > dG_threshold] = np.nan
        y_data[y_data > dG_threshold] = np.nan
        
        #Calculate ddGs with data below limit deleted
        ddG_with_thr = y_data.subtract(x_data)        
        above_limit = ~ddG_with_thr.isnull()
        n_above_limit = above_limit.sum()
        n_compared_list.append(n_above_limit)
        
        r_pearson = x_data.corr(y_data)
        r_sq = r_pearson**2
        textstr = 'n=%.0f\nn$_{thr}$ =%.0f\n$\Delta\Delta G=%.2f$\n$\mathrm{r}=%.2f$' % (data_points_total,n_above_limit,ddG_avg, r_sq)
        axs[0].text(-9, -10, textstr, fontsize=7,
        verticalalignment='top')
        x_ddG = [low_lim + ddG_avg, dG_threshold]
        x_new = [low_lim, dG_threshold - ddG_avg]
        axs[0].plot(x_new,x_ddG,'--r',linewidth = 3)
        counter = 0
figure_counter += 1        
#fig.savefig('/Users/Steve/Desktop/Tecto_temp_figures/single_mutant_profiles_3_' + str(figure_counter) + '.svg')
# PLot bar plot for figure 3
d = {'ddG_median':ddG_median_list,'ddG_std':ddG_std_list,'n':n_compared_list}
ddG_median_df = pd.DataFrame(index=data_sM_11ntR.index,data = d)
plt.bar(range(0,34),ddG_median_df.ddG_median,yerr=ddG_median_df.ddG_std)
plt.ylim([-1,6])
#plt.savefig('/Users/Steve/Desktop/Tecto_temp_figures/ddG_11ntR_sM_allGAAA.svg')

#%% Plot barplot again, but this time replace values above limits by the limits
#THIS IS THE CORRECT PLOT AS OF 09/12/2018
columns_to_plot = data_sM_11ntR.columns[0:50]
counter = -1
ddG_median_list = []
ddG_std_list = []
n_compared_list = []
n_list= []
n_above_limit_list = []

#if greater than limit then replace with limit for each individual!!!!

for receptor in data_sM_11ntR.index[:-1]:
    counter += 1
    
    #Take receptor data; replace values above threshold by threshold
    y_data = data_sM_11ntR.loc[receptor][columns_to_plot].dropna()
    y_data[y_data > dG_threshold] = dG_threshold
    
    #Take WT data and replace values above threshold by threshold
    x_data = data_sM_11ntR.loc['UAUGG_CCUAAG'][columns_to_plot]
    x_data[x_data > dG_threshold] = dG_threshold
    x_data = x_data.dropna()
    
    #These are the scaffolds that have data in both receptor and WT
    scaffolds_to_compare = list(set(y_data.index) & set(x_data.index))
    y_data = y_data.reindex(scaffolds_to_compare)
    x_data = x_data.reindex(scaffolds_to_compare) 
    
    #This is so that we can count the number of values that were above the ddG limit
    limits = dG_threshold - x_data 
    
    #calculate_median and number of datapoints
    ddG = y_data.subtract(x_data)
    print(len(x_data))
    
    
    ddG_above_limit = ~(ddG >= limits)
 #   ddG[~ddG_above_limit] = limits
    
    
    n_above_limit = sum(ddG_above_limit)
    n_above_limit_list.append(n_above_limit)
    ddG_median = ddG.median()
    ddG_std = ddG.std()
    ddG_median_list.append(ddG_median)
    ddG_std_list.append(ddG_std)
    mask = ~((x_data.isna())|(y_data.isna()))
    n = sum(mask)
    n_list.append(n)
    
    x_data[x_data > dG_threshold] = np.nan
    y_data[y_data > dG_threshold] = np.nan
    mask = ~((x_data.isna())|(y_data.isna()))
    n_compared = sum(mask)
    n_compared_list.append(n_compared)
ddG_median_list.append(0)
ddG_std_list.append(0)
n_compared_list.append(np.nan)
n_list.append(np.nan)
n_above_limit_list.append(np.nan)

d = {'ddG_median':ddG_median_list,'ddG_std':ddG_std_list,'n_above_thr':n_compared_list,'n':n_list, 'n_above_limit_list':n_above_limit_list}
ddG_median_df = pd.DataFrame(index=data_sM_11ntR.index,data = d)
plt.bar(range(0,34),ddG_median_df.ddG_median,yerr=ddG_median_df.ddG_std)
plt.ylim([-1,6])
plt.xlim(-1,34)
#%% Compare the specificity of each single mutant
variant_GUAA = data_sM_11ntR.loc['UAUGG_CCUAAG'][150:200]
variant_GUAA.index = all_scaffolds
variant_GAAA = data_sM_11ntR.loc['UAUGG_CCUAAG'][0:50]
variant_GAAA.index = all_scaffolds

specificity = variant_GAAA.subtract(variant_GUAA)


columns_to_plot_GAAA = data_sM_11ntR.columns[0:50]
columns_to_plot_GUAA = data_sM_11ntR.columns[150:200]
counter = -1
spec_median_list = [] #for specificity
spec_std_list = []
n_compared_list = []
n_list= []
n_above_limit_list = []

#if greater than limit then replace with limit for each individual!!!!

for receptor in data_sM_11ntR.index:
    counter += 1
    
    #Take receptor data; replace values above threshold by threshold
    y_data = data_sM_11ntR.loc[receptor][columns_to_plot_GUAA]
    y_data.index = all_scaffolds
    y_data = y_data.dropna()
    y_data[y_data > dG_threshold] = dG_threshold
    
    #Take WT data and replace values above threshold by threshold
    x_data = data_sM_11ntR.loc[receptor][columns_to_plot_GAAA]
    x_data.index = all_scaffolds
    x_data[x_data > dG_threshold] = dG_threshold
    x_data = x_data.dropna()
    
    #These are the scaffolds that have data in both receptor and WT
    scaffolds_to_compare = list(set(y_data.index) & set(x_data.index))
    y_data = y_data.reindex(scaffolds_to_compare)
    x_data = x_data.reindex(scaffolds_to_compare) 
    
    #This is so that we can count the number of values that were above the ddG limit
    limits = dG_threshold - x_data 
    
    #calculate_median and number of datapoints
    ddG = y_data.subtract(x_data)
    print(len(x_data))
    
    
    ddG_above_limit = ~(ddG >= limits)
 #   ddG[~ddG_above_limit] = limits
    
    
    n_above_limit = sum(ddG_above_limit)
    n_above_limit_list.append(n_above_limit)
    ddG_median = ddG.median()
    ddG_std = ddG.std()
    spec_median_list.append(ddG_median)
    spec_std_list.append(ddG_std)
    mask = ~((x_data.isna())|(y_data.isna()))
    n = sum(mask)
    n_list.append(n)
    
    x_data[x_data > dG_threshold] = np.nan
    y_data[y_data > dG_threshold] = np.nan
    mask = ~((x_data.isna())|(y_data.isna()))
    n_compared = sum(mask)
    n_compared_list.append(n_compared)
#spec_median_list.append(0)
#spec_std_list.append(0)
#n_compared_list.append(np.nan)
#n_list.append(np.nan)
#n_above_limit_list.append(np.nan)

d = {'specificity_median':spec_median_list,'specificity_std':spec_std_list,'n_above_thr':n_compared_list,'n':n_list, 'n_above_limit_list':n_above_limit_list}
specificity_median_df = pd.DataFrame(index=data_sM_11ntR.index,data = d)
plt.bar(range(0,34),specificity_median_df.specificity_median,yerr=specificity_median_df.specificity_std)
plt.ylim([-1,6])
plt.xlim(-1,34)
#%% Write a function that calculates bootstraps pearson correlation coefficients
#and 95% confidence interval 
def pearson_bootsrap (x_data,y_data,n_boostraps):
    #x_data is dataframe indexed by 'old_idx' usually containing WT data
    #y_data is dataframe equally indexed, usually containing mut data
    #x_data and y_data containg 2 columns first one has dGs and the second one
    #has the errors. 
    #n_bootsraps is number of iterations for bootstras
    combined_data = pd.DataFrame(index= x_data.index)
    combined_data['x'] = x_data[x_data.columns[0]]
    combined_data['y'] = y_data[y_data.columns[0]]
    combined_data['x_err'] = x_data[x_data.columns[1]]
    combined_data['y_err'] = y_data[y_data.columns[1]]
    mask = ~(combined_data['x'].isna() | combined_data['y'].isna())
    combined_data = combined_data[mask]
    n_comb = len(combined_data)
    list_corr = []
    for i in range(n_bootstraps):
        x = np.random.normal(combined_data['x'],combined_data['x_err'])
        y = np.random.normal(combined_data['y'],combined_data['y_err'])
        r_bs = np.corrcoef(x,y)[0,1]   
        list_corr.append(r_bs)
    list_corr = np.array(list_corr)
    list_corr.sort()
    mean_r_bs = list_corr.mean()
    median_r_bs = np.median(list_corr)
    low_r_bs = list_corr[int(0.05 * n_bootstraps) -1]
    high_r_bs = list_corr[-1*(int(0.05 * n_bootstraps) -1)]
    return (mean_r_bs,median_r_bs,low_r_bs,high_r_bs,n_comb)
#%%Function to calculate Pearson correlations with confidence interval without
#taking into account errors     
def Pearson_and_confidence(x_data,y_data):
    
    r = x_data[x_data.columns[0]].corr(y_data[y_data.columns[0]])
    n_compare = sum(~(x_data[x_data.columns[0]].isna() | y_data[y_data.columns[0]].isna()))
    #calculate confidence interval without taking into account error in each measurement
    
    #in normal space
    z = math.log((1 + r)/(1-r)) * 0.5
    
    #Standard error
    if n_compare > 3:
        SE = 1/math.sqrt(n_compare - 3)
    else:
        SE = np.nan
    
    #interval in normal space
    low_z = z - (1.96 * SE)
    high_z = z + (1.96 * SE)
    
    #95% confidence interval
    low_r = (math.exp(2 * low_z) - 1)/(math.exp(2 * low_z) + 1)
    high_r = (math.exp(2 * high_z) - 1)/(math.exp(2 * high_z) + 1)
    return (r,low_r,high_r,n_compare)
#%%Calculate pearson coeffs for all single mutants at 30 mM Mg
R_pearson_stats = pd.DataFrame(index = data_sM_11ntR.index)
dG_thr = dG_threshold
mean_r_bs_list = []
median_r_bs_list = []
low_r_bs_list = []
high_r_bs_list = []
n_list = []
mean_dG_list = []
ddG_median_list = []
ddG_error_list = []

r_list = []
low_r_list = []
high_r_list = []
n_compare_list = []
n_bootstraps = 1000
WT_data_error = pd.concat([WT_data,WT_error,],axis=1)
x_data = WT_data_error.loc[WT_data_error.index[0:50]].copy()
x_data.columns = ['dG','error']
for receptors in data_sM_11ntR.index:
    #get data for mutant
    mut_data = data_sM_11ntR.loc[receptors]
    mut_error = error_sM_11ntR.loc[receptors]
    mut_data_error = pd.concat([mut_data,mut_error],axis=1)
    y_data = mut_data_error.loc[mut_data_error.index[0:50]].copy()
    y_data.columns = ['dG','error']
    #CALCULATE DDG_MEDIAN BEFORE APPLYING THRESHOLD
    ddG = y_data['dG'].subtract(x_data['dG'])
    ddG_median = ddG.median()
    ddG_error = ddG.std()
#    if ddG_median > 4:
#        ddG_median = 4
    ddG_median_list.append(ddG_median)
    ddG_error_list.append(ddG_error)
    
    
    ##########
    y_data[y_data['dG']>dG_thr] = np.nan
    ######
    x_data[x_data['dG'] >dG_thr] = np.nan
    ######
    ##########
    
    mean_dG_list.append(y_data['dG'].mean())
    ##########
    if len(y_data.dropna()) > 2:
        mean_r_bs,median_r_bs,low_r_bs,high_r_bs,n_comb = pearson_bootsrap(x_data,y_data,n_bootstraps)
        mean_r_bs_list.append(mean_r_bs)
        median_r_bs_list.append(median_r_bs)
        low_r_bs_list.append(low_r_bs)
        high_r_bs_list.append(high_r_bs)
        n_list.append(n_comb)
        
        r,low_r,high_r,n_compare = Pearson_and_confidence(x_data,y_data)
        r_list.append(r)
        low_r_list.append(low_r)
        high_r_list.append(high_r)
        n_compare_list.append(n_compare)
    else:
        mean_r_bs_list.append(np.nan)
        median_r_bs_list.append(np.nan)
        low_r_bs_list.append(np.nan)
        high_r_bs_list.append(np.nan)
        n_list.append(len(y_data.dropna()))
        
        r_list.append(np.nan)
        low_r_list.append(np.nan)
        high_r_list.append(np.nan)
        n_compare_list.append(len(y_data.dropna()))
R_pearson_stats['n_compared'] = n_list
R_pearson_stats['mean_dG'] = mean_dG_list
R_pearson_stats['bootstrap_r_mean'] = mean_r_bs_list
R_pearson_stats['bootstrap_r_median'] = median_r_bs_list
R_pearson_stats['bootstrap_r_low'] = low_r_bs_list
R_pearson_stats['bootstrap_r_high'] = high_r_bs_list

R_pearson_stats['r_pearson'] = r_list
R_pearson_stats['r_low'] = low_r_list
R_pearson_stats['r_high'] = high_r_list
R_pearson_stats['n_comp'] = n_compare_list
R_pearson_stats['ddG'] = ddG_median_list
R_pearson_stats['ddG_error'] = ddG_error_list
#%%Calculate pearson coeffs for all single mutants at 30 mM Mg
#also do it for_comparing 30mM WT to 5 mM mutants
R_pearson_stats_30WT_5Mut = pd.DataFrame(index = data_sM_11ntR.index)
dG_thr = dG_threshold
mean_r_bs_list = []
median_r_bs_list = []
low_r_bs_list = []
high_r_bs_list = []
n_list = []
mean_dG_list = []
ddG_median_list = []
ddG_error_list = []

r_list = []
low_r_list = []
high_r_list = []
n_compare_list = []
n_bootstraps = 1000
WT_data_error = pd.concat([WT_data,WT_error,],axis=1)
x_data = WT_data_error.loc[WT_data_error.index[0:50]].copy()
x_data.columns = ['dG','error']
for receptors in data_sM_11ntR.index:
    #get data for mutant
    mut_data = data_sM_11ntR.loc[receptors]
    mut_error = error_sM_11ntR.loc[receptors]
    mut_data_error = pd.concat([mut_data,mut_error],axis=1)
    y_data = mut_data_error.loc[mut_data_error.index[50:100]].copy()
    y_data.columns = ['dG','error']
    #CALCULATE DDG_MEDIAN BEFORE APPLYING THRESHOLD
    
    x_data.index = all_scaffolds
    y_data.index = all_scaffolds
    
    ddG = y_data['dG'].subtract(x_data['dG'])
    ddG_median = ddG.median()
    ddG_error = ddG.std()
#    if ddG_median > 4:
#        ddG_median = 4
    ddG_median_list.append(ddG_median)
    ddG_error_list.append(ddG_error)
    
    
    ##########
    y_data[y_data['dG']>dG_thr] = np.nan
    ######
    x_data[x_data['dG'] >dG_thr] = np.nan
    ######
    ##########
    
    mean_dG_list.append(y_data['dG'].mean())
    ##########
    if len(y_data.dropna()) > 2:
        mean_r_bs,median_r_bs,low_r_bs,high_r_bs,n_comb = pearson_bootsrap(x_data,y_data,n_bootstraps)
        mean_r_bs_list.append(mean_r_bs)
        median_r_bs_list.append(median_r_bs)
        low_r_bs_list.append(low_r_bs)
        high_r_bs_list.append(high_r_bs)
        n_list.append(n_comb)
        
        r,low_r,high_r,n_compare = Pearson_and_confidence(x_data,y_data)
        r_list.append(r)
        low_r_list.append(low_r)
        high_r_list.append(high_r)
        n_compare_list.append(n_compare)
    else:
        mean_r_bs_list.append(np.nan)
        median_r_bs_list.append(np.nan)
        low_r_bs_list.append(np.nan)
        high_r_bs_list.append(np.nan)
        n_list.append(len(y_data.dropna()))
        
        r_list.append(np.nan)
        low_r_list.append(np.nan)
        high_r_list.append(np.nan)
        n_compare_list.append(len(y_data.dropna()))
R_pearson_stats_30WT_5Mut['n_compared'] = n_list
R_pearson_stats_30WT_5Mut['mean_dG'] = mean_dG_list
R_pearson_stats_30WT_5Mut['bootstrap_r_mean'] = mean_r_bs_list
R_pearson_stats_30WT_5Mut['bootstrap_r_median'] = median_r_bs_list
R_pearson_stats_30WT_5Mut['bootstrap_r_low'] = low_r_bs_list
R_pearson_stats_30WT_5Mut['bootstrap_r_high'] = high_r_bs_list

R_pearson_stats_30WT_5Mut['r_pearson'] = r_list
R_pearson_stats_30WT_5Mut['r_low'] = low_r_list
R_pearson_stats_30WT_5Mut['r_high'] = high_r_list
R_pearson_stats_30WT_5Mut['ddG'] = ddG_median_list
R_pearson_stats_30WT_5Mut['ddG_error'] = ddG_error_list
#%%
colorsl = ['purple','purple','purple','yellow','yellow','yellow','orange','orange','orange',
           'yellow','yellow','yellow','red','red','red','red','red','red','yellow','yellow','yellow',
           'yellow','yellow','yellow','green','green','green','green','green','green','purple','purple',
           'purple','black']
#%% Plot using bootstrap
a = R_pearson_stats['ddG']
b = R_pearson_stats['bootstrap_r_mean']
plt.scatter(a,b,s=120,edgecolors='k',marker='s',c=colorsl)
plt.ylim(-1.2,1.2)
plt.xlim(0,5)
plt.show()
c = [1,3,2,1]
high_error = R_pearson_stats['bootstrap_r_high'].subtract(R_pearson_stats['bootstrap_r_mean'])
low_error = R_pearson_stats['bootstrap_r_mean'].subtract(R_pearson_stats['bootstrap_r_low'])
plt.errorbar(a,b,yerr=[low_error,high_error], linestyle="None",color='k',capsize=3)
plt.show()

b = R_pearson_stats['r_pearson']
#plt.scatter(a,b,s=80,edgecolors='k',marker='^',c=colorsl)
plt.show()

#plot using conventional pearson
plt.figure()
a = R_pearson_stats['ddG']
b = R_pearson_stats['r_pearson']
plt.scatter(a,b,s=120,edgecolors='k',marker='s',c=colorsl)
plt.ylim(-1.2,1.2)
plt.xlim(0,5)
plt.show()
c = [1,3,2,1]
high_error = R_pearson_stats['r_high'].subtract(R_pearson_stats['r_pearson'])
low_error = R_pearson_stats['r_pearson'].subtract(R_pearson_stats['r_low'])
plt.errorbar(a,b,yerr=[low_error,high_error], linestyle="None",color='k',capsize=3)
plt.show()
plt.scatter(a,b,s=120,edgecolors='k',marker='s',c=colorsl)
plt.ylim(-1.2,1.2)
plt.xlim(0,5)
plt.show()
#%% Plot using bootstrap for 30 mM WT vs 5 mM mut
a = R_pearson_stats_30WT_5Mut['ddG']
b = R_pearson_stats_30WT_5Mut['bootstrap_r_mean']
plt.scatter(a,b,s=120,edgecolors='k',marker='s',c=colorsl)
plt.ylim(-1.2,1.2)
plt.xlim(0,5)
plt.show()
c = [1,3,2,1]
high_error = R_pearson_stats_30WT_5Mut['bootstrap_r_high'].subtract(R_pearson_stats_30WT_5Mut['bootstrap_r_mean'])
low_error = R_pearson_stats_30WT_5Mut['bootstrap_r_mean'].subtract(R_pearson_stats_30WT_5Mut['bootstrap_r_low'])
plt.errorbar(a,b,yerr=[low_error,high_error], linestyle="None",color='k',capsize=3)
plt.show()

b = R_pearson_stats_30WT_5Mut['r_pearson']
#plt.scatter(a,b,s=80,edgecolors='k',marker='^',c=colorsl)
plt.show()

#plot using conventional pearson
plt.figure()
a = R_pearson_stats_30WT_5Mut['ddG']
b = R_pearson_stats_30WT_5Mut['r_pearson']
plt.scatter(a,b,s=120,edgecolors='k',marker='o',c=colorsl)
plt.ylim(-1.2,1.2)
plt.xlim(0,5)
plt.show()
c = [1,3,2,1]
high_error = R_pearson_stats_30WT_5Mut['r_high'].subtract(R_pearson_stats_30WT_5Mut['r_pearson'])
low_error = R_pearson_stats_30WT_5Mut['r_pearson'].subtract(R_pearson_stats_30WT_5Mut['r_low'])
plt.errorbar(a,b,yerr=[low_error,high_error], linestyle="None",color='k',capsize=3)
plt.show()

#plot using conventional pearson
plt.figure()
a = R_pearson_stats['ddG']
b = R_pearson_stats['r_pearson']
plt.scatter(a,b,s=120,edgecolors='k',marker='o',c=colorsl)
plt.ylim(-1.2,1.2)
plt.xlim(0,5)
plt.show()
c = [1,3,2,1]
high_error = R_pearson_stats['r_high'].subtract(R_pearson_stats['r_pearson'])
low_error = R_pearson_stats['r_pearson'].subtract(R_pearson_stats['r_low'])
plt.errorbar(a,b,yerr=[low_error,high_error], linestyle="None",color='k',capsize=3)
plt.show()

a = R_pearson_stats_30WT_5Mut['ddG']
b = R_pearson_stats_30WT_5Mut['r_pearson']
plt.scatter(a,b,s=120,edgecolors='k',marker='s',c=colorsl)
plt.ylim(-1.2,1.2)
plt.xlim(0,5)
plt.show()
c = [1,3,2,1]
high_error = R_pearson_stats_30WT_5Mut['r_high'].subtract(R_pearson_stats_30WT_5Mut['r_pearson'])
low_error = R_pearson_stats_30WT_5Mut['r_pearson'].subtract(R_pearson_stats_30WT_5Mut['r_low'])
plt.errorbar(a,b,yerr=[low_error,high_error], linestyle="None",color='k',capsize=3)
plt.show()
#%% Look at mutants with outliers that may need to be investigated further
mut_U7A = all_11ntR[(all_11ntR['r_seq'] == 'AAUGG_CCUAAG')]
mut_U7A = mut_U7A.set_index('old_idx')
mut_U7A = mut_U7A.reindex(all_scaffolds)

plt.scatter(WT_original_data[WT_original_data.length == 11]['dG_Mut2_GAAA'],mut_U7A[mut_U7A.length == 11]['dG_Mut2_GAAA'])
plt.xlim(-10,-7)
plt.ylim(-10,-7)


mut_A5U = all_11ntR[(all_11ntR['r_seq'] == 'UAUGG_CCUAUG')]
mut_A5U = mut_A5U.set_index('old_idx')
mut_A5U = mut_A5U.reindex(all_scaffolds)

plt.figure()
plt.scatter(WT_original_data[WT_original_data.length == 11]['dG_Mut2_GAAA'],mut_A5U[mut_A5U.length == 11]['dG_Mut2_GAAA'])
plt.xlim(-14,-7)
plt.ylim(-14,-7)

#SCAFFOLD 9385 IS ACTING WEIRD
#%%
scaff_9534 = all_11ntR[all_11ntR.old_idx == 9534]
scaff_9534_sM = scaff_9534[scaff_9534.no_mutations == 1]
seq_9534 = scaff_9534_sM.seq

# & (all_data['b_name'] == 'normal')]
#%% Plot with error bars 

#THIS IS NOT COMPLETE YET
x_subplots = 3
y_subplots = 3
fig, axs = plt.subplots(x_subplots, y_subplots, sharey=True, sharex = True, tight_layout=True)
axs = axs.ravel()
counter = -1
figure_counter = 0
for receptors in data_sM_11ntR.index:
    counter += 1
    if counter < (x_subplots * y_subplots):
        #plot 30 mM GAAA
        axs[counter].errorbar(WT_data[columns_to_plot[0:50]],data_sM_11ntR.loc[receptors][columns_to_plot[0:50]],fmt='o',xerr = WT_error[columns_to_plot[0:50]],yerr = error_sM_11ntR.loc[receptors][columns_to_plot[0:50]],ecolor = 'k',mfc = Colors ,mec = 'k')#s=120,edgecolors='k',c=list(Colors.values))
        #plot 5 mM GAAA
        axs[counter].errorbar(WT_data[columns_to_plot[50:100]],data_sM_11ntR.loc[receptors][columns_to_plot[50:100]],fmt='o')#s=120,edgecolors='k',c=list(Colors.values),marker='s')
        #plot 5 mM + 150K
        axs[counter].errorbar(WT_data[columns_to_plot[100:150]],data_sM_11ntR.loc[receptors][columns_to_plot[100:150]],fmt='o')#s=120,edgecolors='k',c=list(Colors.values),marker ='*')
        axs[counter].plot(x,x,':k')
        axs[counter].plot(x,y_thres,':k',linewidth = 0.5)
        axs[counter].plot(y_thres,x,':k',linewidth = 0.5)
        axs[counter].set_xlim(low_lim,high_lim)
        axs[counter].set_ylim(low_lim,high_lim)
        axs[counter].set_xticks(list(range(-14,-4,4)))
        axs[counter].set_yticks(list(range(-14,-4,4)))
        axs[counter].set_title(receptors)
        print(str(receptors))
        x_data = WT_data[columns_to_plot].copy()
        y_data = data_sM_11ntR.loc[receptors][columns_to_plot].copy()
        with_data = ~((x_data.isna()) | (y_data.isna()))
        data_points_total = with_data.sum() 
        x_data[x_data > dG_threshold] = np.nan
        y_data[y_data > dG_threshold] = np.nan
        ddG = y_data.subtract(x_data)
        above_limit = ~ddG.isnull()
        n_above_limit = above_limit.sum()
        ddG_avg = ddG.mean()
        ddG_std = ddG.std()
        r_pearson = x_data.corr(y_data)
        r_sq = r_pearson**2
        textstr = 'n=%.0f\nn$_{thr}$ =%.0f\n$\Delta\Delta G=%.2f$\n$\mathrm{r}=%.2f$' % (data_points_total,n_above_limit,ddG_avg, r_sq)
        axs[counter].text(-9, -10, textstr, fontsize=7,
        verticalalignment='top')
        x_ddG = [ddG_avg + x[0],ddG_avg + x[1]]
        axs[counter].plot(x,x_ddG,'--r',linewidth = 3)
        
    else:
        figure_counter += 1
 #       fig.savefig('/Volumes/NO NAME/Clustermaps/single_mutant_profiles_3_' + str(figure_counter) + '.pdf')
        fig, axs = plt.subplots(x_subplots, y_subplots, sharey=True, sharex = True, tight_layout=True)
        axs = axs.ravel()
                #plot 30 mM GAAA
        axs[0].errorbar(WT_data[columns_to_plot[0:50]],data_sM_11ntR.loc[receptors][columns_to_plot[0:50]],fmt='o')#s=120,edgecolors='k',c=list(Colors.values))
        #plot 5 mM GAAA
        axs[0].errorbar(WT_data[columns_to_plot[50:100]],data_sM_11ntR.loc[receptors][columns_to_plot[50:100]],fmt='o')#s=120,edgecolors='k',c=list(Colors.values),marker='s')
        #plot 5 mM + 150K
        axs[0].errorbar(WT_data[columns_to_plot[100:150]],data_sM_11ntR.loc[receptors][columns_to_plot[100:150]],fmt='o')#s=120,edgecolors='k',c=list(Colors.values),marker ='*')
        axs[0].plot(x,x,':k')
        axs[0].plot(x,y_thres,':k',linewidth = 0.5)
        axs[0].plot(y_thres,x,':k',linewidth = 0.5)
        axs[0].set_xlim(low_lim,high_lim)
        axs[0].set_ylim(low_lim,high_lim)
        axs[0].set_xticks(list(range(-14,-4,4)))
        axs[0].set_yticks(list(range(-14,-4,4)))
        axs[0].set_title(receptors)
        print(str(receptors))
        x_data = WT_data[columns_to_plot].copy()
        y_data = data_sM_11ntR.loc[receptors][columns_to_plot].copy()
        with_data = ~((x_data.isna()) | (y_data.isna()))
        data_points_total = with_data.sum() 
        x_data[x_data > dG_threshold] = np.nan
        y_data[y_data > dG_threshold] = np.nan
        ddG = y_data.subtract(x_data)
        above_limit = ~ddG.isnull()
        n_above_limit = above_limit.sum()
        ddG_avg = ddG.mean()
        ddG_std = ddG.std()
        r_pearson = x_data.corr(y_data)
        r_sq = r_pearson**2
        textstr = 'n=%.0f\nn$_{thr}$ =%.0f\n$\Delta\Delta G=%.2f$\n$\mathrm{r}=%.2f$' % (data_points_total,n_above_limit,ddG_avg, r_sq)
        axs[0].text(-9, -10, textstr, fontsize=7,
        verticalalignment='top')
        x_ddG = [ddG_avg + x[0],ddG_avg + x[1]]
        axs[0].plot(x,x_ddG,'--r',linewidth = 3)
        counter = 0
figure_counter += 1        
#fig.savefig('/Volumes/NO NAME/Clustermaps/single_mutant_profiles_3_' + str(figure_counter) + '.pdf')
#%% PLOT FIGURES SEPARATELY
for receptors in data_sM_11ntR.index:
    plt.figure()
    #plot 30 mM GAAA
    plt.scatter(WT_data[columns_to_plot[0:50]],data_sM_11ntR.loc[receptors][columns_to_plot[0:50]],s=120,edgecolors='k',c=list(Colors.values))
    #plot 5 mM GAAA
    plt.scatter(WT_data[columns_to_plot[50:100]],data_sM_11ntR.loc[receptors][columns_to_plot[50:100]],s=120,edgecolors='k',c=list(Colors.values),marker='s')
    #plot 5 mM + 150K
    plt.scatter(WT_data[columns_to_plot[100:150]],data_sM_11ntR.loc[receptors][columns_to_plot[100:150]],s=120,edgecolors='k',c=list(Colors.values),marker ='*')
    plt.plot(x,x,':k')
    plt.plot(x,y_thres,':k',linewidth = 0.5)
    plt.plot(y_thres,x,':k',linewidth = 0.5)
    plt.xlim(low_lim,high_lim)
    plt.ylim(low_lim,high_lim)
    plt.xticks(list(range(-14,-4,2)))
    plt.yticks(list(range(-14,-4,2)))
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.axes().set_aspect('equal')
    plt.xlabel('$\Delta$G$^{11ntR}_{bind}$ (kcal/mol)',fontsize=22)
    plt.ylabel('$\Delta$G$^{mut}_{bind}$ (kcal/mol)',fontsize=22)
    
    #calculate statistics (with values above limits set to NaN)
    x_data = WT_data[columns_to_plot].copy()
    y_data = data_sM_11ntR.loc[receptors].copy()
    x_data[x_data > dG_threshold] = np.nan
    y_data[y_data > dG_threshold] = np.nan
    ddG = y_data.subtract(x_data)
    ddG_avg = ddG.mean()
    ddG_std = ddG.std()
    r_pearson = x_data.corr(y_data)
    r_sq = r_pearson**2
    textstr = '$\Delta\Delta G=%.2f$\n$\mathrm{r}=%.2f$' % (ddG_avg, r_sq)
    plt.text(-9, -12, textstr, fontsize=7,
    verticalalignment='top')    
#%%PLOT Specific RECEPTOR
high_lim = -6.5
receptors = 'UAUGG_CCUAAG'
plt.figure()
plt.scatter(WT_data[columns_to_plot[0:50]],data_sM_11ntR.loc[receptors][columns_to_plot[0:50]],s=200,edgecolors='k',c=list(Colors.values))
#plot 5 mM GAAA
plt.scatter(WT_data[columns_to_plot[50:100]],data_sM_11ntR.loc[receptors][columns_to_plot[50:100]],s=200,edgecolors='k',c=list(Colors.values),marker='s')
#plot 5 mM + 150K
plt.scatter(WT_data[columns_to_plot[100:150]],data_sM_11ntR.loc[receptors][columns_to_plot[100:150]],s=200,edgecolors='k',c=list(Colors.values),marker ='*')
plt.plot(x,x,':k')
plt.plot(x,y_thres,':k',linewidth = 0.5)
plt.plot(y_thres,x,':k',linewidth = 0.5)
plt.xlim(low_lim,high_lim)
plt.ylim(low_lim,high_lim)
plt.xticks(list(range(-14,-4,2)))
plt.yticks(list(range(-14,-4,2)))
plt.title(receptors)
print(str(receptors))
x_data = WT_data[columns_to_plot].copy()
y_data = data_sM_11ntR.loc[receptors][columns_to_plot].copy()
with_data = ~((x_data.isna()) | (y_data.isna()))
data_points_total = with_data.sum() 
ddG = y_data.subtract(x_data)

 #Calculate mediad ddG with all data
ddG_avg = ddG.median()
ddG_std = ddG.std()
#Get rid of data below limit to calculate correlations
x_data[x_data > dG_threshold] = np.nan
y_data[y_data > dG_threshold] = np.nan

#Calculate ddGs with data below limit deleted
ddG_with_thr = y_data.subtract(x_data)        
above_limit = ~ddG_with_thr.isnull()
n_above_limit = above_limit.sum()
#n_compared_list.append(n_above_limit)
r_pearson = x_data.corr(y_data)
r_sq = r_pearson**2
textstr = 'n=%.0f\nn$_{thr}$ =%.0f\n$\Delta\Delta G=%.2f$\n$\mathrm{r}=%.2f$' % (data_points_total,n_above_limit,ddG_avg, r_sq)
plt.text(-9, -10, textstr, fontsize=7,
verticalalignment='top')
x_ddG = [low_lim + ddG_avg, dG_threshold]
x_new = [low_lim, dG_threshold - ddG_avg]
plt.plot(x_new,x_ddG,'--r',linewidth = 3)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta$G$^{11ntR}_{bind}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta$G$^{mut}_{bind}$ (kcal/mol)',fontsize=22)
#%% Plot profiles with respect to the length of the scaffold
#For GAAA at 30 mM Mg

plot_tandem = True
xlim = [7.5,11.5]
ylim = [-12,-6.5]
plt.plot()
counter = -1
median_per_length = pd.DataFrame(index = data_sM_11ntR.index,columns = ['8','9','10','11'])
for receptors in data_sM_11ntR.index:
    counter += 1
    data_receptor = pd.DataFrame(index=data_sM_11ntR.columns[0:50])
    data_receptor['dG'] = data_sM_11ntR.loc[receptors][0:50]
    data_receptor['length'] = Lengths.values
    A = data_receptor.groupby(by='length')
    if len(data_receptor.dropna()) > 5:
        length_l = []
        median_l = []
        for name, group in A:
            dG = group['dG'].copy()
            dG[dG>dG_threshold] = dG_threshold
            dG_median = dG.median()
#            if dG_median > dG_thr:
#                dG_median = dG_thr
            dG_std = dG.std()
            length_to_plot = group['length'].median() + random.uniform(-0.07,0.07)
            plt.scatter(length_to_plot,dG_median,c = colorsl[counter],s=60,marker='o',edgecolor = 'k')
#            plt.errorbar(group['length'].median(),dG_median,yerr = dG_std,ecolor='k')
            length_l.append(length_to_plot)
            median_l.append(dG_median)
        plt.plot(length_l,median_l,'grey',linewidth=0.75,linestyle='dashed')
        median_per_length.loc[receptors] = median_l
 
plt.plot(xlim,[dG_thr,dG_thr],':k')
plt.xlim(xlim[0],xlim[1])
plt.ylim(ylim[0],ylim[1])
plt.scatter([8,9,10,11],median_per_length.median(),marker = 's', s=120)

if plot_tandem:
    CC_GG = sublib0_grouped.get_group('tandem CC/GG')
    length_l = []
    median_l = [] 
    A = CC_GG.groupby(by = 'length')
    for name, group in A:
        dG = group['dG_Mut2_GAAA'].copy()
        dG[dG>dG_threshold] = dG_threshold
        dG_median = dG.median()
#            if dG_median > dG_thr:
#                dG_median = dG_thr
        dG_std = dG.std()
        length_to_plot = group['length'].median() + random.uniform(-0.07,0.07)
        plt.scatter(length_to_plot,dG_median,c = colorsl[counter],s=120,marker='^',edgecolor = 'k')
#            plt.errorbar(group['length'].median(),dG_median,yerr = dG_std,ecolor='k')
        length_l.append(length_to_plot)
        median_l.append(dG_median)
    plt.plot(length_l,median_l,'black',linewidth=1.5,linestyle='dashed')
    median_per_length.loc[receptors] = median_l
#%% Plot tandem
xlim = [7.5,11.5]
ylim = [-12,-6.5]
for receptor in tandem_receptors:
    next_receptor = sublib0_grouped.get_group(receptor)
    A = next_receptor.groupby('length')
    length_l = []
    median_l= []
    for name, group in A:
        dG_median = group['dG_Mut2_GAAA'].median()
        if dG_median > dG_thr:
            dG_median = dG_thr
        dG_std = group['dG_Mut2_GAAA'].std() 
        length_to_plot = name + random.uniform(-0.07,0.07)
        plt.scatter(length_to_plot,dG_median,c = 'magenta',s=180,marker='o',edgecolor = 'k')
        length_l.append(length_to_plot)
        median_l.append(dG_median)
    plt.plot(length_l,median_l,'grey',linewidth=0.5,linestyle='dashed') 

plt.plot(xlim,[dG_thr,dG_thr],':k')
plt.xlim(xlim[0],xlim[1])
plt.ylim(ylim[0],ylim[1])
plt.scatter([8,9,10,11],median_per_length.median(),marker = 's', s=120)
#%%
plt.figure()
for receptor in tandem_receptors:
    next_receptor = sublib0_grouped.get_group(receptor)
    A = next_receptor.groupby('length')
    length_l = []
    median_l= []
    for name, group in A:
        dG_median = group['dG_Mut2_GUAA_1'].median()
        if dG_median > dG_thr:
            dG_median = dG_thr
        dG_std = group['dG_Mut2_GUAA_1'].std()
        print (dG_median)
        print (dG_std)
        print (name)  
        length_to_plot = name + random.uniform(-0.07,0.07)
        print(length_to_plot)
        plt.scatter(length_to_plot,dG_median,c = 'magenta',s=180,marker='o',edgecolor = 'k')
        length_l.append(length_to_plot)
        median_l.append(dG_median)
    plt.plot(length_l,median_l,'grey',linewidth=0.5,linestyle='dashed') 
    
plt.plot(xlim,[dG_thr,dG_thr],':k')
plt.xlim(xlim[0],xlim[1])
plt.ylim(ylim[0],ylim[1])
plt.scatter([8,9,10,11],median_per_length.median(),marker = 's', s=120)    
#%%
control = 'tandem CC/GG'
control = sublib0_grouped.get_group(control)   
A = control.groupby('length')
length_l = []
median_l= []
for name, group in A:
    dG_median = group['dG_Mut2_GUAA_1'].median()
    if dG_median > dG_thr:
        dG_median = dG_thr
    dG_std = group['dG_Mut2_GUAA_1'].std()
    print (dG_median)
    print (dG_std)
    print (name)  
    length_to_plot = name + random.uniform(-0.07,0.07)
    print(length_to_plot)
    plt.scatter(length_to_plot,dG_median,c = 'magenta',s=180,marker='o',edgecolor = 'k')
    length_l.append(length_to_plot)
    median_l.append(dG_median)
plt.plot(length_l,median_l,'grey',linewidth=0.5,linestyle='dashed') 
plt.plot(xlim,[dG_thr,dG_thr],':k')
plt.xlim(xlim[0],xlim[1])
plt.ylim(ylim[0],ylim[1])
#%% Plot profiles with respect to the length of the scaffold
#For GUAA at 30 mM Mg

plot_tandem = True
xlim = [7.5,11.5]
ylim = [-12,-6.5]
plt.plot()
counter = -1
median_per_length = pd.DataFrame(index = data_sM_11ntR.index,columns = ['8','9','10','11'])
for receptors in data_sM_11ntR.index:
    counter += 1
    data_receptor = pd.DataFrame(index=data_sM_11ntR.columns[150:200])
    data_receptor['dG'] = data_sM_11ntR.loc[receptors][150:200]
    data_receptor['length'] = Lengths.values
    A = data_receptor.groupby(by='length')
    if len(data_receptor.dropna()) > 5:
        length_l = []
        median_l = []
        for name, group in A:
            dG = group['dG'].copy()
            dG[dG>dG_threshold] = dG_threshold
            dG_median = dG.median()
#            if dG_median > dG_thr:
#                dG_median = dG_thr
            dG_std = dG.std()
            length_to_plot = group['length'].median() + random.uniform(-0.07,0.07)
            plt.scatter(length_to_plot,dG_median,c = colorsl[counter],s=60,marker='o',edgecolor = 'k')
#            plt.errorbar(group['length'].median(),dG_median,yerr = dG_std,ecolor='k')
            length_l.append(length_to_plot)
            median_l.append(dG_median)
        plt.plot(length_l,median_l,'grey',linewidth=0.75,linestyle='dashed')
        median_per_length.loc[receptors] = median_l


plt.plot(xlim,[dG_thr,dG_thr],':k')
plt.xlim(xlim[0],xlim[1])
plt.ylim(ylim[0],ylim[1])
plt.scatter([8,9,10,11],median_per_length.median(),marker = 's', s=120)

if plot_tandem:
    CC_GG = sublib0_grouped.get_group('tandem CC/GG')
    length_l = []
    median_l = [] 
    A = CC_GG.groupby(by = 'length')
    for name, group in A:
        dG = group['dG_Mut2_GUAA_1'].copy()
        dG[dG>dG_threshold] = dG_threshold
        dG_median = dG.median()
#            if dG_median > dG_thr:
#                dG_median = dG_thr
        dG_std = dG.std()
        length_to_plot = group['length'].median() + random.uniform(-0.07,0.07)
        plt.scatter(length_to_plot,dG_median,c = colorsl[counter],s=120,marker='^',edgecolor = 'k')
#            plt.errorbar(group['length'].median(),dG_median,yerr = dG_std,ecolor='k')
        length_l.append(length_to_plot)
        median_l.append(dG_median)
    plt.plot(length_l,median_l,'black',linewidth=1.5,linestyle='dashed')
    median_per_length.loc[receptors] = median_l
#%%Compare double mutants to check if binding to base pairs
canonical = all_11ntR[all_11ntR.r_seq == 'UAUGG_CCUAAG'].copy()
canonical = canonical[canonical.b_name == 'normal']
canonical = canonical.set_index('old_idx')

double_mutants = all_11ntR[all_11ntR.no_mutations == 2].copy()
double_mutants = double_mutants[double_mutants.b_name == 'normal']
double_mutants_group = double_mutants.groupby(by ='r_seq')

triple_mutants = all_11ntR[all_11ntR.no_mutations == 3].copy()
triple_mutants = triple_mutants[triple_mutants.b_name == 'normal']
triple_mutants_group = triple_mutants.groupby(by ='r_seq')

variant = 'UAGGG_CCCAAG'
variant_data = double_mutants_group.get_group(variant)
variant_data = variant_data.set_index('old_idx')
GAAA_data = variant_data['dG_Mut2_GAAA']
GUAA_data = variant_data['dG_Mut2_GUAA_1']
print(GAAA_data)
print(GUAA_data)

ddG_GAAA = -1 * canonical['dG_Mut2_GAAA'].subtract(GAAA_data).median()
print(ddG_GAAA)
ddG_GUAA = -1 * canonical['dG_Mut2_GUAA_1'].subtract(GUAA_data).median()
print(ddG_GUAA)

#%%
variant = 'UUAGG_CCUAAG'
variant_data = all_data[all_data.r_seq == variant]
variant_data = variant_data.set_index('old_idx')

canonical = 'UAUGG_CCUAAG'
canonical_data = all_data[(all_data.r_seq == canonical) & (all_data.b_name == 'normal')]
canonical_data = canonical_data.set_index('old_idx')

GAAA_data = variant_data['dG_Mut2_GAAA'].copy()
GAAA_data[GAAA_data > dG_threshold] = dG_threshold

GUAA_data = variant_data['dG_Mut2_GUAA_1'].copy()
GUAA_data[GUAA_data > dG_threshold] = dG_threshold

ddG_GAAA = GAAA_data.subtract(canonical_data['dG_Mut2_GAAA'])
ddG_GAAA_median = ddG_GAAA.median()
ddG_GAAA_std = ddG_GAAA.std()

ddG_GUAA = GUAA_data.subtract(canonical_data['dG_Mut2_GUAA_1'])
ddG_GUAA_median = ddG_GUAA.median()
ddG_GUAA_std = ddG_GUAA.std()

plt.bar([1,2],[ddG_GAAA_median,ddG_GUAA_median],yerr=[ddG_GAAA_std,ddG_GUAA_std])
plt.ylim([-1,6])
plt.xlim(0,3)

#%%
A = variant_data.groupby('length')
for name, group in A:
    print(name)
    print(len(group))
    print(group['dG_Mut2_GAAA'])
    print(group['dG_Mut2_GAAA'].median())
    
for name, group in A:
    print(name)
    print(len(group))
    print(group['dG_Mut2_GUAA_1'])
    print(group['dG_Mut2_GUAA_1'].median())    
#%% Analyze GAAA vs GUAA
#Barplots    
columns_to_plot = data_sM_11ntR.columns[150:200]
counter = -1
ddG_median_list = []
ddG_std_list = []
n_compared_list = []
n_list= []

for receptor in data_sM_11ntR.index[:-1]:
    counter += 1
    y_data = data_sM_11ntR.loc[receptor][columns_to_plot]
    x_data = data_sM_11ntR.loc['UAUGG_CCUAAG'][columns_to_plot]
    plt.scatter(x_data,y_data,c=colorsl[counter])
    #calculate_median and number of datapoints
    ddG = y_data.subtract(x_data)
    ddG_median = ddG.median()
    ddG_std = ddG.std()
    ddG_median_list.append(ddG_median)
    ddG_std_list.append(ddG_std)
    mask = ~((x_data.isna())|(y_data.isna()))
    n = sum(mask)
    n_list.append(n)
    
    x_data[x_data > dG_threshold] = np.nan
    y_data[y_data > dG_threshold] = np.nan
    mask = ~((x_data.isna())|(y_data.isna()))
    n_compared = sum(mask)
    n_compared_list.append(n_compared)
ddG_median_list.append(0)
ddG_std_list.append(0)
n_compared_list.append(np.nan)
n_list.append(np.nan)

d = {'ddG_median':ddG_median_list,'ddG_std':ddG_std_list,'n_above_thr':n_compared_list,'n':n_list}
ddG_median_df = pd.DataFrame(index=data_sM_11ntR.index,data = d)
plt.bar(range(0,34),ddG_median_df.ddG_median,yerr=ddG_median_df.ddG_std)
plt.ylim([-1,2])
plt.xlim(-1,34)
#%% COMPARE SMs to WT with GUAA----- BARPLOT
#THIS IS CORRECT as of 09/12/2018
columns_to_plot = data_sM_11ntR.columns[150:155]
counter = -1
ddG_median_list = []
ddG_std_list = []
n_compared_list = []
n_list= []
n_above_limit_list = []

#if greater than limit then replace with limit for each individual!!!!

for receptor in data_sM_11ntR.index[:-1]:
    counter += 1
    
    #Take receptor data; replace values above threshold by threshold
    y_data = data_sM_11ntR.loc[receptor][columns_to_plot].dropna()
    y_data[y_data > dG_threshold] = dG_threshold
    
    #Take WT data and replace values above threshold by threshold
    x_data = data_sM_11ntR.loc['UAUGG_CCUAAG'][columns_to_plot]
    x_data[x_data > dG_threshold] = dG_threshold
    x_data = x_data.dropna()
    
    #These are the scaffolds that have data in both receptor and WT
    scaffolds_to_compare = list(set(y_data.index) & set(x_data.index))
    y_data = y_data.reindex(scaffolds_to_compare)
    x_data = x_data.reindex(scaffolds_to_compare) 
    
    #This is so that we can count the number of values that were above the ddG limit
    limits = dG_threshold - x_data 
    
    #calculate_median and number of datapoints
    ddG = y_data.subtract(x_data)
    print(len(x_data))
    
    
    ddG_above_limit = ~(ddG >= limits)
 #   ddG[~ddG_above_limit] = limits
    
    
    n_above_limit = sum(ddG_above_limit)
    n_above_limit_list.append(n_above_limit)
    ddG_median = ddG.median()
    ddG_std = ddG.std()
    ddG_median_list.append(ddG_median)
    ddG_std_list.append(ddG_std)
    mask = ~((x_data.isna())|(y_data.isna()))
    n = sum(mask)
    n_list.append(n)
    
    x_data[x_data > dG_threshold] = np.nan
    y_data[y_data > dG_threshold] = np.nan
    mask = ~((x_data.isna())|(y_data.isna()))
    n_compared = sum(mask)
    n_compared_list.append(n_compared)
ddG_median_list.append(0)
ddG_std_list.append(0)
n_compared_list.append(np.nan)
n_list.append(np.nan)
n_above_limit_list.append(np.nan)

d = {'ddG_median':ddG_median_list,'ddG_std':ddG_std_list,'n_above_thr':n_compared_list,'n':n_list, 'n_above_limit_list':n_above_limit_list}
ddG_median_df = pd.DataFrame(index=data_sM_11ntR.index,data = d)
plt.bar(range(0,34),ddG_median_df.ddG_median,yerr=ddG_median_df.ddG_std)
plt.ylim([-1,2])
plt.xlim(-1,34)
#%%
#comparing 5mM with and without potassium
apply_thr = True
n_list = []
diff_list = []
std_list = []
for receptors in data_sM_11ntR.index:
    receptor_5mM = data_sM_11ntR.loc[receptors][50:100].copy()
    receptor_5mM.index = all_scaffolds
    receptor_5mM150K = data_sM_11ntR.loc[receptors][100:150].copy()
    receptor_5mM150K.index = all_scaffolds
    if apply_thr:
        receptor_5mM[receptor_5mM > dG_thr] = np.nan
        receptor_5mM150K[receptor_5mM150K > dG_thr] = np.nan
    ddG_pot = receptor_5mM150K.subtract(receptor_5mM)
    diff_avg = ddG_pot.mean()
    diff_std = ddG_pot.std()
    n = len(ddG_pot.dropna())
    diff_list.append(diff_avg)
    n_list.append(n)
    std_list.append(diff_std)
ddG_w_wo_K = pd.DataFrame(index = data_sM_11ntR.index)
ddG_w_wo_K['ddG'] = diff_list
ddG_w_wo_K['std'] = std_list
ddG_w_wo_K['n'] = n_list

plt.figure()
plt.bar(range(0,34),ddG_w_wo_K['ddG'],yerr= ddG_w_wo_K['std'])
#%%5mM vs 5mM + 150K Analysis
#Heat map|
data_5mM = data_sM_11ntR[data_sM_11ntR.columns[50:100]].copy()
data_5mM.columns = all_scaffolds
#data_5mM = data_5mM.reindex(['UAUGG_CCUAAG','space1','space2'] + list(data_5mM.index[:-1]))

data_5mM_150K = data_sM_11ntR[data_sM_11ntR.columns[100:150]].copy()
data_5mM_150K.columns = all_scaffolds
#data_5mM_150K = data_5mM_150K.reindex(['UAUGG_CCUAAG','space1','space2'] + list(data_5mM_150K.index[:-1]))

mask = (data_5mM > dG_thr) | (data_5mM_150K > dG_thr)
data_5mM[data_5mM > dG_thr] = np.nan
data_5mM_150K[data_5mM_150K > dG_thr] = np.nan
diff = data_5mM_150K.subtract(data_5mM)

sns.heatmap(diff,vmin=-2,vmax=2, cmap='coolwarm')
#sns.heatmap(mask,mask=mask,cbar=False) 
#linewidths = 0.03,linecolor = 'grey')
#%%
plt.figure(figsize = (4,20))
plt.scatter(ddG_w_wo_K['ddG'],-1 * np.linspace(0,100,34),c=colorsl,edgecolor='black',s=40,marker='s')
plt.errorbar(ddG_w_wo_K['ddG'],-1 * np.linspace(0,100,34),xerr=ddG_w_wo_K['std'],linestyle="None",color='k',capsize=3)
plt.xlim(-2,2)
plt.savefig('/Users/Steve/Desktop/Tecto_temp_figures/test_5.svg')
#from matplotlib.patches import Rectangle
#g.add_patch(Rectangle((3,4),1,1, fill = False, edgecolor = 'black', lw = 0.5))
#%%
x = np.random.randn(10, 10)
sns.heatmap(x, annot=True)
sns.heatmap(x, mask=x < 1, cbar=False,
            annot=True, annot_kws={"weight": "bold"})
#%%
#FILL NANS BECAUSE STUPID ILLUSTRATROR
ddG_w_wo_K[ddG_w_wo_K.isna()] = 0.5
plt.bar(range(0,34),ddG_w_wo_K['ddG'],yerr= ddG_w_wo_K['std'])
plt.ylim([-1.6,1.6])
#plt.savefig('/Users/Steve/Desktop/Tecto_temp_figures/potassium_11ntR_sM.svg')
#%%
#%% Plot without error bars
# 5 mM Mg vs 5 mM + 150K
ddG_median_list = []
ddG_std_list = []
n_compared_list = []
x_subplots = 3
y_subplots = 3
fig, axs = plt.subplots(x_subplots, y_subplots, sharey=True, sharex = True, tight_layout=True)
axs = axs.ravel()
counter = -1
figure_counter = 0
for receptors in data_sM_11ntR.index:
    counter += 1
    if counter < (x_subplots * y_subplots):
        axs[counter].scatter(data_sM_11ntR.loc[receptors][50:100],data_sM_11ntR.loc[receptors][100:150],s=120,edgecolors='k',c=list(Colors.values),marker ='*')
        axs[counter].plot(x,x,':k')
        axs[counter].plot(x,y_thres,':k',linewidth = 0.5)
        axs[counter].plot(y_thres,x,':k',linewidth = 0.5)
        axs[counter].set_xlim(low_lim,high_lim)
        axs[counter].set_ylim(low_lim,high_lim)
        axs[counter].set_xticks(list(range(-14,-4,4)))
        axs[counter].set_yticks(list(range(-14,-4,4)))
        axs[counter].set_title(receptors)
        print(str(receptors))
        x_data = data_sM_11ntR.loc[receptors][50:100].copy()
        x_data.index = all_scaffolds
        y_data = data_sM_11ntR.loc[receptors][100:150].copy()
        y_data.index = all_scaffolds
        with_data = ~((x_data.isna()) | (y_data.isna()))
        data_points_total = with_data.sum() 
        ddG = y_data.subtract(x_data)
        
        #Calculate mediad ddG with all data
        ddG_avg = ddG.median()
        ddG_median_list.append(ddG_avg)
        ddG_std = ddG.std()
        ddG_std_list.append(ddG_std)
        
        #Get rid of data below limit to calculate correlations
        x_data[x_data > dG_threshold] = np.nan
        y_data[y_data > dG_threshold] = np.nan
        
        #Calculate ddGs with data below limit deleted
        ddG_with_thr = y_data.subtract(x_data)        
        above_limit = ~ddG_with_thr.isnull()
        n_above_limit = above_limit.sum()
        n_compared_list.append(n_above_limit)
        
        r_pearson = x_data.corr(y_data)
        r_sq = r_pearson**2
        textstr = 'n=%.0f\nn$_{thr}$ =%.0f\n$\Delta\Delta G=%.2f$\n$\mathrm{r}=%.2f$' % (data_points_total,n_above_limit,ddG_avg, r_sq)
        axs[counter].text(-9, -10, textstr, fontsize=7,
        verticalalignment='top')
        x_ddG = [low_lim + ddG_avg, dG_threshold]
        x_new = [low_lim, dG_threshold - ddG_avg]
        axs[counter].plot(x_new,x_ddG,'--r',linewidth = 3)

        
    else:
        figure_counter += 1
#        fig.savefig('/Users/Steve/Desktop/Tecto_temp_figures/single_mutant_profiles_3_' + str(figure_counter) + '.svg')
        fig, axs = plt.subplots(x_subplots, y_subplots, sharey=True, sharex = True, tight_layout=True)
        axs = axs.ravel()
                #plot 30 mM GAAA
        axs[0].scatter(data_sM_11ntR.loc[receptors][50:100],data_sM_11ntR.loc[receptors][100:150],s=120,edgecolors='k',c=list(Colors.values),marker ='*')        
        axs[0].plot(x,x,':k')
        axs[0].plot(x,y_thres,':k',linewidth = 0.5)
        axs[0].plot(y_thres,x,':k',linewidth = 0.5)
        axs[0].set_xlim(low_lim,high_lim)
        axs[0].set_ylim(low_lim,high_lim)
        axs[0].set_xticks(list(range(-14,-4,4)))
        axs[0].set_yticks(list(range(-14,-4,4)))
        axs[0].set_title(receptors)
        print(str(receptors))
        x_data = data_sM_11ntR.loc[receptors][50:100].copy()
        x_data.index = all_scaffolds
        y_data = data_sM_11ntR.loc[receptors][100:150].copy()
        y_data.index = all_scaffolds
        with_data = ~((x_data.isna()) | (y_data.isna()))
        data_points_total = with_data.sum() 
        ddG = y_data.subtract(x_data)
        
        #Calculate mediad ddG with all data
        ddG_avg = ddG.median()
        ddG_median_list.append(ddG_avg)
        ddG_std = ddG.std()
        ddG_std_list.append(ddG_std)
        
        #Get rid of data below limit to calculate correlations
        x_data[x_data > dG_threshold] = np.nan
        y_data[y_data > dG_threshold] = np.nan
        
        #Calculate ddGs with data below limit deleted
        ddG_with_thr = y_data.subtract(x_data)        
        above_limit = ~ddG_with_thr.isnull()
        n_above_limit = above_limit.sum()
        n_compared_list.append(n_above_limit)
        
        r_pearson = x_data.corr(y_data)
        r_sq = r_pearson**2
        textstr = 'n=%.0f\nn$_{thr}$ =%.0f\n$\Delta\Delta G=%.2f$\n$\mathrm{r}=%.2f$' % (data_points_total,n_above_limit,ddG_avg, r_sq)
        axs[0].text(-9, -10, textstr, fontsize=7,
        verticalalignment='top')
        x_ddG = [low_lim + ddG_avg, dG_threshold]
        x_new = [low_lim, dG_threshold - ddG_avg]
        axs[0].plot(x_new,x_ddG,'--r',linewidth = 3)
        counter = 0
figure_counter += 1        
#fig.savefig('/Users/Steve/Desktop/Tecto_temp_figures/single_mutant_profiles_3_' + str(figure_counter) + '.svg')

#%%
spec_list = []
plt.figure()
for receptors in data_sM_11ntR.index:
#    receptors = 'UAUGG_CCUACG'
    GAAA_data = data_sM_11ntR.loc[receptors][0:50].copy()
    GAAA_data.index = range(0,50)
    GUAA_data = data_sM_11ntR.loc[receptors][150:200].copy()
    GUAA_data.index = range(0,50)
    spec = GUAA_data.subtract(GAAA_data)
    spec_avg = spec.median()
    spec_list.append(spec_avg)
    plt.scatter(GAAA_data,GUAA_data)
    plt.show()
plt.figure()
plt.bar(range(0,34),spec_list,yerr=ddG_std)
plt.ylim([-1,6])
#%%
WT_GUAA = WT_data[150:200]
WT_GUAA.index = range(0,50)
WT_GUAA[WT_GUAA > -8.1] = np.nan
WT_GUAA = WT_GUAA.dropna()
#%%
spec_list = []
for receptors in data_sM_11ntR.index:
#    receptors = 'UAUGG_CCUACG'
#    plt.figure()
    GAAA_data = data_sM_11ntR.loc[receptors][0:50].copy()
    GAAA_data.index = range(0,50)
    GUAA_data = data_sM_11ntR.loc[receptors][150:200].copy()
    GUAA_data.index = range(0,50)
    GUAA_data[GUAA_data > dG_thr] = np.nan
    spec = GUAA_data.subtract(WT_GUAA)
    spec_avg = spec.median()
    spec_list.append(spec_avg)
#    plt.scatter(WT_GUAA,GUAA_data)
#    plt.xlim(-10,-5)
#    plt.ylim(-10,-5)
#    plt.plot([-10,-5],[-10,-5],'--r')
#    plt.title(receptors)
#    plt.show()
plt.figure()
plt.bar(range(0,34),spec_list,yerr=ddG_std)
plt.ylim([-1,3])
#%%
#Calculate ddGs at 30 mM Mg (ACTUALLY I AM USING ALL_DATA WITH GAAA) relative to WT
data_sM_30Mg = data_sM_11ntR[columns_to_plot].copy()    
data_sM_30Mg[data_sM_30Mg>-7.1] = np.nan
reference = data_sM_30Mg.loc['UAUGG_CCUAAG']
ddG_sM_30Mg = data_sM_30Mg.subtract(reference)
ddG_average = ddG_sM_30Mg.mean(axis=1)
ddG_std = ddG_sM_30Mg.std(axis=1)
plt.figure()
#ddG_average.plot(kind = 'bar', yerr = ddG_std, edgecolor = 'black',error_kw=dict(ecolor='black',elinewidth=0.5),width=1.0,ylim=(-1,6))

plt.bar(range(0,34),ddG_average,yerr=ddG_std)
plt.ylim([-1,6])
plt.savefig('ddG_11ntR_sM_allGAAA.svg')
#%% Plot TL_TLRs 5mM vs 5 mM + potassitum 
for receptors in data_sM_11ntR.index:
    data_5mM = data_sM_11ntR.loc[receptors][50:100]
    data_5mM150K = data_sM_11ntR.loc[receptors][100:150]
    plt.figure()
    plt.scatter(data_5mM,data_5mM150K,s=120,edgecolors='k',c=list(Colors.values))
    plt.plot(x,x,':k')
    plt.plot(x,y_thres,':k',linewidth = 0.5)
    plt.plot(y_thres,x,':k',linewidth = 0.5)
    plt.xlim(low_lim,high_lim)
    plt.ylim(low_lim,high_lim)
    plt.xticks(list(range(-14,-4,2)))
    plt.yticks(list(range(-14,-4,2)))
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.axes().set_aspect('equal')
    plt.xlabel('$\Delta$G$^{11ntR}_{bind}$ (kcal/mol)',fontsize=22)
    plt.ylabel('$\Delta$G$^{mut}_{bind}$ (kcal/mol)',fontsize=22)
