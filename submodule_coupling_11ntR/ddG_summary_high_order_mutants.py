#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:03:05 2018

@author: Steve
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import pearsonr
import math
#%%

'''---------------Import data ------------------'''
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
all_data = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
all_11ntR = pd.read_csv(data_path + 'all_11ntRs_unique.csv' )
all_11ntR = all_11ntR.drop_duplicates('seq')
all_11ntR_normal = all_11ntR[all_11ntR['b_name'] == 'normal']
high_mutants = all_11ntR_normal[all_11ntR_normal['no_mutations'] > 2].copy()
#
#import information about receptors that have been selected previously for analysis
receptors_types_matrix = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/'
                        'initial_clustering_all_data/receptors_types_matrix.pkl')

#from double mutant data previously generated select only those being analyzed further
selected_11ntRs = receptors_types_matrix[receptors_types_matrix['type_11ntR'] == 1]
selected = []
for sequence in high_mutants['r_seq']:
    if sequence in selected_11ntRs.index:
        selected.append(1)
    else:
        selected.append(0)
high_mutants['selected'] = selected
#
high_mutants = high_mutants[high_mutants['selected'] == 1]
high_mutants_seq = list(set(high_mutants['r_seq']))
high_mutants_grouped = high_mutants.groupby('r_seq')
wt_data = all_11ntR_normal[all_11ntR_normal['r_seq'] == 'UAUGG_CCUAAG']
wt_data = wt_data.set_index('old_idx')
#%%
dG_threshold = -7.1

high_order_ddG_summary = pd.DataFrame(index=high_mutants_seq)

ddG_30Mg_list = []
n_30Mg = []
ddG_5Mg_list = []
n_5Mg = []
ddG_5Mg150K_list = []
n_5Mg150K = []
ddG_GUAA_list = []
n_GUAA = []
n_corr_list = []
r_list = []

for sequence in high_order_ddG_summary.index:
    next_data = high_mutants_grouped.get_group(sequence)
    next_data = next_data.set_index('old_idx')
    n_total_30Mg = len(next_data['dG_Mut2_GAAA'].dropna())
    n_total_5Mg = len(next_data['dG_Mut2_GAAA_5mM_2'].dropna())
    n_total_5Mg150K = len(next_data['dG_Mut2_GAAA_5mM_150mMK_1'].dropna())
    n_total_GUAA = len(next_data['dG_Mut2_GUAA_1'].dropna())
    
    next_data = next_data.reindex(wt_data.index)
    ddG_30Mg = next_data['dG_Mut2_GAAA'] - wt_data['dG_Mut2_GAAA']
    ddG_5Mg = next_data['dG_Mut2_GAAA_5mM_2'] - wt_data['dG_Mut2_GAAA_5mM_2']
    ddG_5Mg150K = next_data['dG_Mut2_GAAA_5mM_150mMK_1'] - wt_data['dG_Mut2_GAAA_5mM_150mMK_1']
    ddG_GUAA = next_data['dG_Mut2_GUAA_1'] - wt_data['dG_Mut2_GUAA_1']
    
    ddG_30Mg_list.append(ddG_30Mg.median())
    ddG_5Mg_list.append(ddG_5Mg.median())
    ddG_5Mg150K_list.append(ddG_5Mg150K.median())
    ddG_GUAA_list.append(ddG_GUAA.median())
    
    n_30Mg.append(n_total_30Mg)
    n_5Mg.append(n_total_5Mg)
    n_5Mg150K.append(n_total_5Mg150K)
    n_GUAA.append(n_total_GUAA)
    
    
    #calculate r only for 30mM Mg relative to wt
    x_data = wt_data['dG_Mut2_GAAA'].copy()
    x_data[x_data > dG_threshold] = np.nan
    y_data = next_data['dG_Mut2_GAAA'].copy()
    y_data[y_data > dG_threshold] = np.nan
    data = pd.concat([x_data,y_data],axis = 1)
    data.columns = ['c1','c2']
    data = data.dropna()
    n_corr = len(data)
    n_corr_list.append(n_corr)
    r = data.corr()
    r = r['c1']['c2']
    r_list.append(r)
#%%    
    
high_order_ddG_summary['ddG_30Mg'] =  ddG_30Mg_list
high_order_ddG_summary['n_30Mg'] = n_30Mg
high_order_ddG_summary['ddG_5Mg'] = ddG_5Mg_list
high_order_ddG_summary['n_5Mg'] = n_5Mg
high_order_ddG_summary['ddG_5Mg150K'] = ddG_5Mg150K_list
high_order_ddG_summary['n_5Mg150K'] = n_5Mg150K
high_order_ddG_summary['ddG_GUAA'] = ddG_GUAA_list
high_order_ddG_summary['n_GUAA'] = n_GUAA
high_order_ddG_summary['n_corr'] = n_corr_list
high_order_ddG_summary['r'] = r_list
#%% 
A = pd.DataFrame(0,index = high_order_ddG_summary.index, columns = high_mutants.columns[-14:])

#%%
for sequence in A.index:
    b = high_mutants[high_mutants['r_seq'] == sequence]
    c = b.loc[b.index[0]][high_mutants.columns[-14:]]
    A.loc[sequence] = c
#%%
core_mutations = []
platform_mutations = []
wobble_mutations = []
bulge_mutations = []
WC_mutations = []

for sequence in A.index:
    if A.loc[sequence]['mutations_ 3']:
        bulge_mutations.append(True)
    else:
        bulge_mutations.append(False)
    
    if (A.loc[sequence]['mutations_ 2']) | (A.loc[sequence]['mutations_ 4']) | (A.loc[sequence]['mutations_ 8']) | (A.loc[sequence]['mutations_ 9']):
        core_mutations.append(True)
    else:
        core_mutations.append(False)

    if (A.loc[sequence]['mutations_10']) | (A.loc[sequence]['mutations_11']):
        platform_mutations.append(True)
    else:
        platform_mutations.append(False)

    if (A.loc[sequence]['mutations_ 1']) | (A.loc[sequence]['mutations_12']):
        wobble_mutations.append(True)
    else:
        wobble_mutations.append(False)  
        
    if (A.loc[sequence]['mutations_ 5']) | (A.loc[sequence]['mutations_ 7']):
        WC_mutations.append(True)
    else:
        WC_mutations.append(False)    
    
#%%        
A['core_mutations'] = core_mutations
A['platform_mutations'] = platform_mutations
A['wobble_mutations'] = wobble_mutations
A['bulge_mutations'] = bulge_mutations
A['WC_mutations'] = WC_mutations   

#%%
high_order_ddG_summary = pd.concat([high_order_ddG_summary,A],axis = 1)


#%%
filt_data = high_order_ddG_summary[high_order_ddG_summary['n_corr'] > 5]
plt.scatter(filt_data['ddG_30Mg'],filt_data['r'])

#%%
#import double mutant summaries
double_mutant_ddG_summary = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/'
                                          'TL_TLR_CHIP_Data/submodule_coupling_11ntR/'
                                          'double_mutant_ddG_summary_pred_3.pkl') 
#%%

#%% Plot DOUBLE MUTANTS
#Plot r vs ddG 
n_threshold = 2

data_filt = double_mutant_ddG_summary[double_mutant_ddG_summary['n_corr'] > n_threshold]
#data_filt = data_filt[data_filt['core_mutations'] == False]
y_data = data_filt['r']
x_data = data_filt['ddG_30Mg']
plt.scatter(x_data,y_data, s = 120, color = 'blue', edgecolor = 'black')

outlier = data_filt[(data_filt['r'] < 0.2) & (data_filt['core_mutations'] == False)]
print('outlier : ' + outlier.index)


data_filt = data_filt[data_filt['core_mutations']]
y_data = data_filt['r']
x_data = data_filt['ddG_30Mg']
plt.scatter(x_data,y_data, s = 120, color = 'yellow', edgecolor = 'black')
plt.xlim([-1,5])
plt.plot([-1,5],[1,1],'--k')
plt.plot([-1,5],[0,0],'--k')
#%PLOT HIGHER ORDER MUTANTS
data_filt = high_order_ddG_summary[high_order_ddG_summary['n_corr'] > n_threshold]
y_data = data_filt['r']
x_data = data_filt['ddG_30Mg']
plt.scatter(x_data,y_data, s = 120, marker = 's', color = 'blue', edgecolor = 'black')

outlier = data_filt[(data_filt['r'] < 0.2) & (data_filt['core_mutations'] == False)]
print('outlier : ' + outlier.index)


data_filt = data_filt[data_filt['core_mutations']]
y_data = data_filt['r']
x_data = data_filt['ddG_30Mg']
plt.scatter(x_data,y_data, s = 120, marker = 's',color = 'yellow', edgecolor = 'black')
plt.xlim([-1,5])
plt.plot([-1,5],[1,1],'--k')
plt.plot([-1,5],[0,0],'--k')

#%% Plot DOUBLE MUTANTS
#Plot r vs ddG 
n_threshold = 2

for n in list(range(3,51)):


    data_filt = double_mutant_ddG_summary[double_mutant_ddG_summary['n_corr'] > n_threshold]
    to_plot = data_filt[data_filt['n_corr'] == n]
    y_data = to_plot['r']
    x_data = to_plot['ddG_30Mg']
    plt.scatter(x_data,y_data, s = 2 * n + 20, color = 'blue', edgecolor = 'black')
    
    data_filt = data_filt[data_filt['core_mutations']]
    to_plot = data_filt[data_filt['n_corr'] == n]
    y_data = to_plot['r']
    x_data = to_plot['ddG_30Mg']
    plt.scatter(x_data,y_data, s = 2* n + 20, color = 'yellow', edgecolor = 'black')
    plt.xlim([-1,5])
    plt.plot([-1,5],[1,1],'--k')
    plt.plot([-1,5],[0,0],'--k')
    
    
    #%PLOT HIGHER ORDER MUTANTS
    data_filt = high_order_ddG_summary[high_order_ddG_summary['n_corr'] > n_threshold]
    to_plot = data_filt[data_filt['n_corr'] == n]
    y_data = to_plot['r']
    x_data = to_plot['ddG_30Mg']
    plt.scatter(x_data,y_data, s = 2 * n + 20, marker = 's', color = 'blue', edgecolor = 'black')

    
    
    data_filt = data_filt[data_filt['core_mutations']]
    to_plot = data_filt[data_filt['n_corr'] == n]
    y_data = to_plot['r']
    x_data = to_plot['ddG_30Mg']
    plt.scatter(x_data,y_data, s = 2 * n + 20, marker = 's',color = 'yellow', edgecolor = 'black')
    plt.xlim([-1,5])
    plt.plot([-1,5],[1,1],'--k')
    plt.plot([-1,5],[0,0],'--k')

plt.xlabel('$\Delta\Delta$G$^{avg}_{bind}$ (kcal/mol)',fontsize=12)
plt.ylabel('Pearson r',fontsize=12)

#%%
#get the 'best variants' i.e. the ones with the highest correlation coefficients 
#at 30 mM Mg, and with a high number of datapoints 
#n >40 and r > 0.95

#mutants are compared at 5 mM Mg
#what condition to compare WT 

condition = 'dG_Mut2_GAAA'
#condition = 'dG_Mut2_GAAA_5mM_2'

mask = (double_mutant_ddG_summary['n_corr'] > 40) & (double_mutant_ddG_summary['r'] > 0.95)
good_dMs = double_mutant_ddG_summary[mask].index

mask = (high_order_ddG_summary['n_corr'] > 40) & (high_order_ddG_summary['r'] > 0.95)
good_hMs = high_order_ddG_summary[mask].index

high_corr_variants = list(good_dMs) + list(good_hMs)

wt_data = wt_data.copy()

#
#for the variants that are highly correlated to the WT at 30 mM Mg, calculate the
#ddG at 5mM Mg wrt to WT at 30 mM Mg and also calculate the correlation coefficient

all_11ntR_grouped = all_11ntR_normal.groupby('r_seq')
ddG_avg = []
r_list = []
n_total = []
n_corr = []

for sequences in high_corr_variants:
    next_data = all_11ntR_grouped.get_group(sequences).copy()
    next_data = next_data.set_index('old_idx')
    next_data = next_data.reindex(wt_data.index)
    ddG = next_data['dG_Mut2_GAAA_5mM_2'] - wt_data[condition]
    n_total.append(len(ddG.dropna()))
    ddG_avg.append(ddG.median())
    
    x_data = next_data['dG_Mut2_GAAA_5mM_2'].copy()
    x_data[x_data > dG_threshold] = np.nan
    
    y_data = wt_data[condition].copy()
    y_data[y_data > dG_threshold] = np.nan
    
    data = pd.concat([x_data,y_data],axis = 1)
    data = data.dropna()
    n_corr.append(len(data.dropna()))
    r = data.corr()
    r = r['dG_Mut2_GAAA_5mM_2'][condition]
    r_list.append(r)
    
control_high_corr_mutants = pd.DataFrame(index = high_corr_variants)
control_high_corr_mutants['ddG_5mM_vs_30mMWT'] = ddG_avg
control_high_corr_mutants['n_total'] = n_total
control_high_corr_mutants['r'] = r_list
control_high_corr_mutants['n_corr'] = n_corr
#%
#plt.scatter(control_high_corr_mutants['ddG_5mM_vs_30mMWT'],control_high_corr_mutants['r'])
#%
#n_threshold = 2
#filt_data = control_high_corr_mutants[control_high_corr_mutants['n_corr'] > 2]
#plt.scatter(filt_data['ddG_5mM_vs_30mMWT'],filt_data['r'], s =120, edgecolor = 'black')
#plt.xlim([-1,5])
#plt.ylim([-1.2,1.2])
#plt.plot([-1,5],[1,1],'--k')
#plt.plot([-1,5],[0,0],'--k')
#plt.xlabel('$\Delta\Delta$G$^{avg}_{bind}$ (kcal/mol)',fontsize=12)
#plt.ylabel('Pearson r',fontsize=12)


for n in list(range(3,51)):
    filt_data = control_high_corr_mutants[control_high_corr_mutants['n_corr'] > n_threshold]
    to_plot = filt_data[filt_data['n_corr'] == n]
    y_data = to_plot['r']
    x_data = to_plot['ddG_5mM_vs_30mMWT']
    plt.scatter(x_data,y_data, s = 2 * n + 20, color = 'blue', edgecolor = 'black')
    
plt.xlim([-1,5])
#plt.plot([-1,5],[1,1],'--k')
plt.plot([-1,5],[0.95,0.95],'--r')
plt.ylim([-1.2,1.2])
#plt.ylim

#plt.plot([-1,5],[0,0],'--k')    
plt.xlabel('$\Delta\Delta$G$^{avg}_{bind}$ (kcal/mol)',fontsize=12)
plt.ylabel('Pearson r',fontsize=12)

#%% What are the ouliers in the ddG vs r plot above
# Outliers were considered those with r < 0.8 and more stable than 3kcal/mol
#outliers
mask = (double_mutant_ddG_summary['n_corr'] > 2) & (double_mutant_ddG_summary['ddG_30Mg'] < 3) & (double_mutant_ddG_summary['r'] < 0.8)
seqs1 = double_mutant_ddG_summary[mask].index
print(seqs1)

mask = (high_order_ddG_summary['n_corr'] > 2) & (high_order_ddG_summary['ddG_30Mg'] < 3) & (high_order_ddG_summary['r'] < 0.8)
seqs2 = high_order_ddG_summary[mask].index
print(seqs2)

#mask = (high_order_ddG_summary['n_corr'] > 5) & (high_order_ddG_summary['ddG_30Mg'] < 4) & (high_order_ddG_summary['r'] < 0.5)
#seqs = high_order_ddG_summary[mask].index
#print(seqs)

outliers = list(seqs1) + list(seqs2) 
#%%
'''Import Data'''
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
data_11ntR = pd.read_csv(data_path + 'all_11ntRs_unique.csv')
data_11ntR = data_11ntR[data_11ntR.b_name == 'normal']
print('data with normal flaking base pairs is: ', data_11ntR.shape)
entire_lib = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
entire_lib = entire_lib.drop_duplicates(subset='seq')
entire_lib = entire_lib[entire_lib['b_name'] == 'normal']
grouped_lib = entire_lib.groupby('r_seq')

for outlier in outliers:
    alt_TLR_seq = outlier
    color_to_plot = 'navy'
    
    '''Condition to compare'''
    condition = 'dG_Mut2_GAAA'  
    condition2 = 'dG_Mut2_GAAA' 
    
    #'dG_Mut2_GAAA' 
    #'dG_Mut2_GAAA_5mM_150mMK_1'
    
    plot_GUAA = False
    
    '''Limits for plot figures'''
    low_lim = -14
    high_lim = -6
    
    '''Set threshold'''
    dG_threshold = -7.1
    set_threshold = True
    
    '''plot originial or thresholded data'''
    plot_original = True
    
    
    '''WT'''
    # create dataframe with WT receptors and flanking bp == 'normal'
    WT_11ntR = data_11ntR[data_11ntR.r_seq == 'UAUGG_CCUAAG']
    WT_11ntR = grouped_lib.get_group('UAUGG_CCUAAG')
    unique_scaffolds = set(WT_11ntR['old_idx'])
    WT_11ntR = WT_11ntR.set_index('old_idx')
    WT_11ntR = WT_11ntR.reindex(unique_scaffolds)
    #
    alt_TLR = data_11ntR[data_11ntR.r_seq == alt_TLR_seq]
    alt_TLR = grouped_lib.get_group(alt_TLR_seq)
    alt_TLR = alt_TLR.set_index('old_idx')
    alt_TLR = alt_TLR.reindex(unique_scaffolds)
    # Keep original data without thresholding for plotting.
    WT_11ntR_original = WT_11ntR.copy()
    n_wt = len(WT_11ntR_original[condition].dropna())
    alt_TLR_original = alt_TLR.copy()
    n_alt = len(alt_TLR_original[condition].dropna())
    #
    '''Set values above dG threshold to NAN'''
    if set_threshold:
        cond = alt_TLR[condition2].copy()
        cond[cond>dG_threshold] = np.nan
        alt_TLR[condition2] = cond
        del cond
        cond = WT_11ntR[condition].copy()
        cond[cond>dG_threshold] = np.nan
        WT_11ntR[condition] = cond
    #
    #ddG and correlation values are calculated based on thresholfed data
    '''Calculate ddG values'''
    ddG = alt_TLR[condition2] - WT_11ntR[condition] 
    ddG_average = ddG.mean()
    print('ddG_mean using values < threshold: ' + str(ddG_average))
    ddG_std = ddG.std()
    
    
    #calculate ddG median taking into account all values
    ddG_2 = alt_TLR_original[condition2] - WT_11ntR_original[condition]
    ddG_median = ddG_2.median()
    ddG_median_std = ddG_2.std()
    print('ddG_median using all values: ' + str(ddG_median))
    
    
    '''Correlation coefficient of thresholded data'''
    data_comb = pd.concat([alt_TLR[condition2],WT_11ntR[condition]],axis = 1)
    data_comb = data_comb.dropna()
    data_comb.columns = ['column_1','column_2']
    R = data_comb.corr(method = 'pearson')
    R_pearson = R['column_1']['column_2']
    n_corr = len(data_comb)
    
    
    #calculate rmse with respect to the median
    model = WT_11ntR[condition] + ddG_median
    model_df = pd.concat([alt_TLR[condition2],model],axis = 1)
    model_df = model_df.dropna()
    model_df.columns = [['actual','model']]
    rms = sqrt(mean_squared_error(model_df['actual'],model_df['model']))
    '''Correlation coefficient original data'''
    data_comb_original = pd.concat([alt_TLR_original[condition2],WT_11ntR_original[condition]],axis = 1)
    data_comb_original.columns = ['column_1','column_2']
    R_original = data_comb_original.corr(method = 'pearson')
    data_comb_original[data_comb_original>-7.1] = np.nan
    data_clean = data_comb_original.dropna()
    r,p = pearsonr(data_clean['column_1'],data_clean['column_2'])
    #
    #calculate RMSE
    data = alt_TLR[condition].dropna().copy()
    diff = data - ddG_average
    
    #
    #Color based on the length of the CHIP piece 
    Colors = WT_11ntR.length.copy()
    Colors[Colors == 8] = color_to_plot#'red'
    Colors[Colors == 9] = color_to_plot#'blue'
    Colors[Colors == 10] = color_to_plot#'orange'
    Colors[Colors == 11] = color_to_plot#'black'
    
    #
    if plot_original:
       fig1 = plt.figure()
       x = [low_lim, dG_threshold] #for plotting  y= x line
       y_thres = [dG_threshold,dG_threshold]
       x_ddG = [ddG_average + x[0],ddG_average + x[1]]
       plt.plot(x,x_ddG,'--r',linewidth = 3)
       plt.scatter(WT_11ntR_original[condition],alt_TLR_original[condition2],s=120,edgecolors='k',c=Colors,linewidth=2)
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
       plt.title(alt_TLR_seq)
       plt.text(-9.5,-11,'ddG_avg = ' + "%.2f" % round(ddG_median,2))
       plt.text(-9.5,-11.5,'r_pearson = ' + "%.2f" % round(R_pearson,2))
       plt.text(-9.5,-12,'rmse = ' + "%.2f" % round(rms,2))
       plt.text(-9.5,-12.5,'n_total (WT) = ' + "%.0f" % round(n_wt,2)) 
       plt.text(-9.5,-13,'n_total (mutant) = ' + "%.0f" % round(n_alt,2))   
       plt.text(-9.5,-13.5,'n correlated = ' + "%.0f" % round(n_corr,2))   
       fig1.show() 
    else:
        fig1 = plt.figure()
        x = [low_lim, dG_threshold] #for plotting  y= x line
        y_thres = [dG_threshold,dG_threshold]
        x_ddG = [ddG_average + x[0],ddG_average + x[1]]
        plt.plot(x,x_ddG,'--r',linewidth = 3)
        plt.scatter(WT_11ntR[condition],alt_TLR[condition2],s=120,edgecolors='k',c=Colors)
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
        plt.title(alt_TLR_seq)
        fig1.show()

    figure_name = 'wt_vs_' + alt_TLR_seq + '_scatter.svg' 
    fig1.savefig('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/submodule_coupling_11ntR/r_vs_ddG_figures/' + figure_name)

#%% print subplots for specific variants in the r vs ddG plot

  
    
mask = (double_mutant_ddG_summary['n_corr'] > 5) & (double_mutant_ddG_summary['ddG_30Mg'] > 3) & (double_mutant_ddG_summary['r'] < 0.8)
seqs1 = double_mutant_ddG_summary[mask].index
print(seqs1)

mask = (high_order_ddG_summary['n_corr'] > 5) & (high_order_ddG_summary['ddG_30Mg'] > 3) & (high_order_ddG_summary['r'] < 0.8)
seqs2 = high_order_ddG_summary[mask].index
print(seqs2)

#mask = (high_order_ddG_summary['n_corr'] > 5) & (high_order_ddG_summary['ddG_30Mg'] < 4) & (high_order_ddG_summary['r'] < 0.5)
#seqs = high_order_ddG_summary[mask].index
#print(seqs)

outliers = list(seqs1) + list(seqs2)  
number_plot = math.ceil(len(outliers)/4)   
counter = 0
    
    
for each_plot in range(number_plot):   
    fig, axs = plt.subplots(2,2, figsize=(8,8),sharex = True, sharey = True)
    first = each_plot * 4
    axs = axs.ravel()   
    i = -1    
    for outlier in outliers[first:first+3]:
        i += 1
        alt_TLR_seq = outlier
        color_to_plot = 'red'
        
        '''Condition to compare'''
        condition = 'dG_Mut2_GAAA'  
        condition2 = 'dG_Mut2_GAAA' 
        
        #'dG_Mut2_GAAA' 
        #'dG_Mut2_GAAA_5mM_150mMK_1'
        
        plot_GUAA = False
        
        '''Limits for plot figures'''
        low_lim = -14
        high_lim = -6
        
        '''Set threshold'''
        dG_threshold = -7.1
        set_threshold = True
        
        '''plot originial or thresholded data'''
        plot_original = True
        
        
        '''WT'''
        # create dataframe with WT receptors and flanking bp == 'normal'
        WT_11ntR = data_11ntR[data_11ntR.r_seq == 'UAUGG_CCUAAG']
        WT_11ntR = grouped_lib.get_group('UAUGG_CCUAAG')
        unique_scaffolds = set(WT_11ntR['old_idx'])
        WT_11ntR = WT_11ntR.set_index('old_idx')
        WT_11ntR = WT_11ntR.reindex(unique_scaffolds)
        #
        alt_TLR = data_11ntR[data_11ntR.r_seq == alt_TLR_seq]
        alt_TLR = grouped_lib.get_group(alt_TLR_seq)
        alt_TLR = alt_TLR.set_index('old_idx')
        alt_TLR = alt_TLR.reindex(unique_scaffolds)
        # Keep original data without thresholding for plotting.
        WT_11ntR_original = WT_11ntR.copy()
        n_wt = len(WT_11ntR_original[condition].dropna())
        alt_TLR_original = alt_TLR.copy()
        n_alt = len(alt_TLR_original[condition].dropna())
        #
        '''Set values above dG threshold to NAN'''
        if set_threshold:
            cond = alt_TLR[condition2].copy()
            cond[cond>dG_threshold] = np.nan
            alt_TLR[condition2] = cond
            del cond
            cond = WT_11ntR[condition].copy()
            cond[cond>dG_threshold] = np.nan
            WT_11ntR[condition] = cond
        #
        #ddG and correlation values are calculated based on thresholfed data
        '''Calculate ddG values'''
        ddG = alt_TLR[condition2] - WT_11ntR[condition] 
        ddG_average = ddG.mean()
        print('ddG_mean using values < threshold: ' + str(ddG_average))
        ddG_std = ddG.std()
        
        
        #calculate ddG median taking into account all values
        ddG_2 = alt_TLR_original[condition2] - WT_11ntR_original[condition]
        ddG_median = ddG_2.median()
        ddG_median_std = ddG_2.std()
        print('ddG_median using all values: ' + str(ddG_median))
        
        
        '''Correlation coefficient of thresholded data'''
        data_comb = pd.concat([alt_TLR[condition2],WT_11ntR[condition]],axis = 1)
        data_comb = data_comb.dropna()
        data_comb.columns = ['column_1','column_2']
        R = data_comb.corr(method = 'pearson')
        R_pearson = R['column_1']['column_2']
        n_corr = len(data_comb)
        
        
        #calculate rmse with respect to the median
        model = WT_11ntR[condition] + ddG_median
        model_df = pd.concat([alt_TLR[condition2],model],axis = 1)
        model_df = model_df.dropna()
        model_df.columns = [['actual','model']]
        rms = sqrt(mean_squared_error(model_df['actual'],model_df['model']))
        '''Correlation coefficient original data'''
        data_comb_original = pd.concat([alt_TLR_original[condition2],WT_11ntR_original[condition]],axis = 1)
        data_comb_original.columns = ['column_1','column_2']
        R_original = data_comb_original.corr(method = 'pearson')
        data_comb_original[data_comb_original>-7.1] = np.nan
        data_clean = data_comb_original.dropna()
        r,p = pearsonr(data_clean['column_1'],data_clean['column_2'])
        #
        #calculate RMSE
        data = alt_TLR[condition].dropna().copy()
        diff = data - ddG_average
        
        #
        #Color based on the length of the CHIP piece 
        Colors = WT_11ntR.length.copy()
        Colors[Colors == 8] = color_to_plot#'red'
        Colors[Colors == 9] = color_to_plot#'blue'
        Colors[Colors == 10] = color_to_plot#'orange'
        Colors[Colors == 11] = color_to_plot#'black'
        
        #
        if plot_original:
           x = [low_lim, dG_threshold] #for plotting  y= x line
           y_thres = [dG_threshold,dG_threshold]
           x_ddG = [ddG_average + x[0],ddG_average + x[1]]
           axs[i].plot(x,x_ddG,'--r',linewidth = 3)
           axs[i].scatter(WT_11ntR_original[condition],alt_TLR_original[condition2],s=40,edgecolors='k',c=Colors,linewidth=2)
           axs[i].plot(x,x,':k')
           axs[i].plot(x,y_thres,':k',linewidth = 0.5)
           axs[i].plot(y_thres,x,':k',linewidth = 0.5)
           axs[i].set_xlim(low_lim,high_lim)
           axs[i].set_ylim(low_lim,high_lim)
           axs[i].set_xticks(list(range(-14,-4,2)))
           axs[i].set_yticks(list(range(-14,-4,2)))
           axs[i].tick_params(axis='both', which='major', labelsize=12)
    #       axs[i].set_aspect('equal')
    #       axs[i].axes().set_aspect('equal')
    #       axs[i].set_xlabel('$\Delta$G$^{11ntR}_{bind}$ (kcal/mol)',fontsize=12)
    #       axs[i].set_ylabel('$\Delta$G$^{mut}_{bind}$ (kcal/mol)',fontsize=12)
           axs[i].set_title(alt_TLR_seq)
           axs[i].text(-10,-11,'ddG_avg = ' + "%.2f" % round(ddG_median,2))
           axs[i].text(-10,-11.5,'r_pearson = ' + "%.2f" % round(R_pearson,2))
           axs[i].text(-10,-12,'rmse = ' + "%.2f" % round(rms,2))
           axs[i].text(-10,-12.5,'n_tot (WT) = ' + "%.0f" % round(n_wt,2)) 
           axs[i].text(-10,-13,'n_tot (mutant) = ' + "%.0f" % round(n_alt,2))   
           axs[i].text(-10,-13.5,'n corr = ' + "%.0f" % round(n_corr,2))   
    figure_name = 'test'       
    fig.savefig('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/submodule_coupling_11ntR/r_vs_ddG_figures/' + figure_name)





