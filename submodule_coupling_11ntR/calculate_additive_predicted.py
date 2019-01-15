#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:40:50 2018

@author: Steve
"""

import pandas as pd
import numpy as np
#from clustering_functions import get_dG_for_scaffold
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
from sklearn.metrics import mean_squared_error
from math import sqrt
#%%
double_mutant_ddG_summary = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/'
                                          'TL_TLR_CHIP_Data/submodule_coupling_11ntR/'
                                          'double_mutant_ddG_summary.pkl')
single_mutant_ddG_summary = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/'
                                      'TL_TLR_CHIP_Data/submodule_coupling_11ntR/'
                                      'single_mutant_ddG_summary.pkl')

#%%

add_30Mg = []
add_5Mg = []
add_5Mg150K = []
add_GUAA = []
single_mutants_added_list = []

for sequence in double_mutant_ddG_summary.index:
    additive_30Mg = 0
    additive_5Mg = 0
    additive_5Mg150K = 0
    additive_GUAA = 0
    single_mutants_added = []
    for column in double_mutant_ddG_summary.columns[22:34]:
        if double_mutant_ddG_summary.loc[sequence][column] != '-':
            residue = double_mutant_ddG_summary.loc[sequence][column]
            for single_mut in single_mutant_ddG_summary.index:
                if single_mutant_ddG_summary.loc[single_mut][column] == residue:
                    single_mutants_added.append(single_mut)
                    additive_30Mg += single_mutant_ddG_summary.loc[single_mut]['ddG_30Mg']
                    additive_5Mg += single_mutant_ddG_summary.loc[single_mut]['ddG_5Mg']
                    additive_5Mg150K += single_mutant_ddG_summary.loc[single_mut]['ddG_5Mg150K']
                    additive_GUAA += single_mutant_ddG_summary.loc[single_mut]['ddG_GUAA']               
    #append
    add_30Mg.append(additive_30Mg)
    add_5Mg.append(additive_5Mg)
    add_5Mg150K.append(additive_5Mg150K)
    add_GUAA.append(additive_GUAA)
    single_mutants_added_list.append(single_mutants_added)
    
#%%
dM_predict_vs_obs = pd.DataFrame(index = double_mutant_ddG_summary.index)
dM_predict_vs_obs['pred_add_30Mg'] = add_30Mg 
dM_predict_vs_obs['pred_add_5Mg'] = add_5Mg    
dM_predict_vs_obs['pred_add_5Mg150K'] = add_5Mg150K    
dM_predict_vs_obs['pred_add_GUAA'] = add_GUAA  
dM_predict_vs_obs['single_mutants_added'] = single_mutants_added_list    

#%%
double_mutant_ddG_summary = pd.concat([double_mutant_ddG_summary,dM_predict_vs_obs],axis=1)
double_mutant_ddG_summary.to_pickle('/Users/Steve/Desktop/Data_analysis_code/'
                                      'TL_TLR_CHIP_Data/submodule_coupling_11ntR/'
                                      'double_mutant_ddG_summary_with_pred.pkl')
#%%
#calculate r coefficients for double mutants
#calculate only at 30 mM Mg 
dG_thres = -7.1

data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
all_data = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
all_data = all_data.drop_duplicates('seq')
all_data_normal = all_data[all_data['b_name'] == 'normal']
all_data_grouped = all_data_normal.groupby('r_seq')

WT = all_data_normal[all_data_normal['r_seq'] == 'UAUGG_CCUAAG']
WT = WT.set_index('old_idx')
x_data = WT['dG_Mut2_GAAA'].copy()
x_data[x_data > dG_thres] = np.nan

n_corr = []
r_list = []

for sequence in double_mutant_ddG_summary.index:
    next_data = all_data_grouped.get_group(sequence)
    next_data = next_data.set_index('old_idx')
    next_data  =next_data.reindex(x_data.index)
    y_data = next_data['dG_Mut2_GAAA'].copy()
    y_data[y_data > dG_thres] = np.nan
    
    diff = x_data - y_data
    n = len(diff.dropna())
    n_corr.append(n)
    
    r = y_data.corr(x_data)
    r_list.append(r)


double_mutant_ddG_summary['r'] = r_list
double_mutant_ddG_summary['n_corr'] = n_corr
#%%
# NO NEED TO SAVE AGAIN; UNLESS STARTING FROM THE BEGGINING WITH: generate_ddG_summary_sM_and_dM.py
double_mutant_ddG_summary.to_pickle('/Users/Steve/Desktop/Data_analysis_code/'
                                          'TL_TLR_CHIP_Data/submodule_coupling_11ntR/'
                                          'double_mutant_ddG_summary_pred_2.pkl') 

#%%
mask = double_mutant_ddG_summary['n_corr'] > 3
double_mutant_filt = double_mutant_ddG_summary[mask]

plot_lims = [-1,6]
plt.scatter(double_mutant_filt['ddG_5Mg150K'],double_mutant_filt['pred_add_5Mg150K'])
plt.xlim(plot_lims)
plt.ylim(plot_lims)
plt.plot(plot_lims,plot_lims,'--')
#

#%%
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
entire_lib = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
entire_lib = entire_lib.drop_duplicates('seq')
entire_lib = entire_lib[entire_lib['b_name'] == 'normal']
entire_lib_grouped = entire_lib.groupby('r_seq')
wt_data = entire_lib_grouped.get_group('UAUGG_CCUAAG')
wt_data = wt_data.set_index('old_idx')
#%%
#These part of the script plot predicted(additive) vs observed for each of 
#the scaffolds individually

#under which conditions to compare predicted (additive) vs observed 
condition = 'dG_Mut2_GAAA'
#condition = 'dG_Mut2_GAAA_5mM_150mMK_1'
ddG_threshold = 4

submodule1 = 'bulge'
submodule2 = 'platform'


#select mutations
mask = double_mutant_ddG_summary[submodule1 + '_mutations'] & double_mutant_ddG_summary[submodule2 + '_mutations']
selected = double_mutant_ddG_summary[mask]
predicted_all = []
observed_all = []

for sequences in selected.index:
    #these are the single mutants that add to the double mutant
    single_mutants = selected.loc[sequences]['single_mutants_added']
    sM1 = entire_lib_grouped.get_group(single_mutants[0])
    sM1 = sM1.set_index('old_idx')
    sM1 = sM1.reindex(wt_data.index)

    sM2 = entire_lib_grouped.get_group(single_mutants[1])
    sM2 = sM2.set_index('old_idx')
    sM2 = sM2.reindex(wt_data.index)
    
    next_data = entire_lib_grouped.get_group(sequences)
    next_data = next_data.set_index('old_idx')
    next_data = next_data.reindex(wt_data.index)
    #predicted = ddG (wrt WT) for sM1 + ddG (wrt WT) for sM2 
    predicted = (sM1[condition] - wt_data[condition]) + (sM2[condition] - wt_data[condition])
    #observed = ddG (wrt Wt) for double mutant
    observed = (next_data[condition] - wt_data[condition])
    predicted_all.append(predicted.values)
    observed_all.append(observed.values)
#
ax_lim = [-1,5]
predicted = [each for sublist in predicted_all for each in sublist]
observed = [each for sublist in observed_all for each in sublist]
plt.scatter(predicted,observed)
plt.axes().set_aspect('equal')
plt.xlim(ax_lim)
plt.ylim(ax_lim)
plt.plot(ax_lim,ax_lim,'--')
plt.plot(ax_lim,[ddG_threshold,ddG_threshold],'--r')
plt.plot([ddG_threshold,ddG_threshold],ax_lim,'--r')
plt.xlabel('Predicted additive')
plt.ylabel('Observed')
plt.title('Mutations ' + submodule1 + ' vs. ' + submodule2)


#calculate R
values = pd.DataFrame()
values['predicted'] = predicted
values['observed'] = observed
#
#filted values
values2 = values.copy()
values2[values2 > ddG_threshold] = np.nan
values2 = values2.dropna()
r = values2.corr()
r = r['predicted']['observed']
plt.text(3,0,'r = ' + "%.2f" % round(r,2))

#
#thresholded coupling
bins = list(np.linspace(-2.5,2.5,40))
plt.figure()
coupling = values2['observed'] - values2['predicted']
n, bins, patches = plt.hist(coupling,bins, normed = 1)
plt.title('Mutations ' + submodule1 + ' vs. ' + submodule2)
(mu, sigma) = norm.fit(coupling)
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
plt.xlim([-2.5,2.5])
plt.ylim([0,1.3])

plt.text(1,1,'mean = ' + "%.2f" % round(mu,2))
plt.text(1,0.9,'std.dev = ' + "%.2f" % round(sigma,2))

#%%
#These part of the script plot predicted(additive) vs observed AVERAGED ACROSS SCAFFOLDS

#under which conditions to compare predicted (additive) vs observed 
condition = 'dG_Mut2_GAAA'
#condition = 'dG_Mut2_GAAA_5mM_150mMK_1'
ddG_threshold = 4

submodule1 = 'platform'
submodule2 = 'wobble'


#select mutations
mask = double_mutant_ddG_summary[submodule1 + '_mutations'] & double_mutant_ddG_summary[submodule2 + '_mutations']
selected = double_mutant_ddG_summary[mask]
predicted_all = []
observed_all = []

for sequences in selected.index:
    #these are the single mutants that add to the double mutant
    single_mutants = selected.loc[sequences]['single_mutants_added']
    sM1 = entire_lib_grouped.get_group(single_mutants[0])
    sM1 = sM1.set_index('old_idx')
    sM1 = sM1.reindex(wt_data.index)

    sM2 = entire_lib_grouped.get_group(single_mutants[1])
    sM2 = sM2.set_index('old_idx')
    sM2 = sM2.reindex(wt_data.index)
    
    next_data = entire_lib_grouped.get_group(sequences)
    next_data = next_data.set_index('old_idx')
    next_data = next_data.reindex(wt_data.index)
    #predicted = ddG (wrt WT) for sM1 + ddG (wrt WT) for sM2 
    predicted = (sM1[condition] - wt_data[condition]) + (sM2[condition] - wt_data[condition])
    #observed = ddG (wrt Wt) for double mutant
    observed = (next_data[condition] - wt_data[condition])
    predicted_all.append(predicted.median())
    observed_all.append(observed.median())
#
ax_lim = [-1,5]
predicted = predicted_all
observed = observed_all
plt.scatter(predicted,observed, s = 120, color = 'red', edgecolor = 'black')
plt.axes().set_aspect('equal')
plt.xlim(ax_lim)
plt.ylim(ax_lim)
plt.plot(ax_lim,ax_lim,'--')
plt.plot(ax_lim,[ddG_threshold,ddG_threshold],'--r')
plt.plot([ddG_threshold,ddG_threshold],ax_lim,'--r')
plt.xlabel('Predicted additive')
plt.ylabel('Observed')
plt.title('Mutations ' + submodule1 + ' vs. ' + submodule2)


#calculate RMSE only for values that are below a ddG value of 4 kcal/mol
data = pd.DataFrame()
data['observed'] = observed
data['predicted'] = predicted
n_total = len(data)
data[data>4] = np.nan
data = data.dropna()
n_threshold = len(data)
rms = sqrt(mean_squared_error(data['observed'], data['predicted']))
plt.text(2,0,'rmse = ' + "%.2f" % round(rms,2))
plt.text(2,-0.25,'n$_{total}$ = ' + "%.0f" % round(n_total,0))
plt.text(2,-0.50,'n$_{< 4 kcal/mol}$ = ' + "%.0f" % round(n_threshold,0))




#%%
#These part of the script plot predicted(additive) vs observed for each of 
#the scaffolds individually
#FOR ALL OF THE SUBMODULES SIMULTANEOUSLY

submodules = ['WC','bulge','platform','wobble']
for submodule1 in submodules:
    for submodule2 in submodules:
        if submodule2 != submodule1:
            #under which conditions to compare predicted (additive) vs observed 
            #condition = 'dG_Mut2_GAAA'
            condition = 'dG_Mut2_GAAA_5mM_150mMK_1'
            ddG_threshold = 4
        
            #select mutations
            mask = double_mutant_ddG_summary[submodule1 + '_mutations'] & double_mutant_ddG_summary[submodule2 + '_mutations']
            selected = double_mutant_ddG_summary[mask]
            predicted_all = []
            observed_all = []
            
            for sequences in selected.index:
                #these are the single mutants that add to the double mutant
                single_mutants = selected.loc[sequences]['single_mutants_added']
                sM1 = entire_lib_grouped.get_group(single_mutants[0])
                sM1 = sM1.set_index('old_idx')
                sM1 = sM1.reindex(wt_data.index)
            
                sM2 = entire_lib_grouped.get_group(single_mutants[1])
                sM2 = sM2.set_index('old_idx')
                sM2 = sM2.reindex(wt_data.index)
                
                next_data = entire_lib_grouped.get_group(sequences)
                next_data = next_data.set_index('old_idx')
                next_data = next_data.reindex(wt_data.index)
                #predicted = ddG (wrt WT) for sM1 + ddG (wrt WT) for sM2 
                predicted = (sM1[condition] - wt_data[condition]) + (sM2[condition] - wt_data[condition])
                #observed = ddG (wrt Wt) for double mutant
                observed = (next_data[condition] - wt_data[condition])
                predicted_all.append(predicted.values)
                observed_all.append(observed.values)
            #
            plt.figure()
            ax_lim = [-1,5]
            predicted = [each for sublist in predicted_all for each in sublist]
            observed = [each for sublist in observed_all for each in sublist]
            plt.scatter(predicted,observed)
            plt.axes().set_aspect('equal')
            plt.xlim(ax_lim)
            plt.ylim(ax_lim)
            plt.plot(ax_lim,ax_lim,'--')
            plt.plot(ax_lim,[ddG_threshold,ddG_threshold],'--r')
            plt.plot([ddG_threshold,ddG_threshold],ax_lim,'--r')
            plt.xlabel('Predicted additive')
            plt.ylabel('Observed')
            plt.title('Mutations ' + submodule1 + ' vs. ' + submodule2)
            
            
            #calculate R
            values = pd.DataFrame()
            values['predicted'] = predicted
            values['observed'] = observed
            #
            #filted values
            values2 = values.copy()
            values2[values2 > ddG_threshold] = np.nan
            values2 = values2.dropna()
            r = values2.corr()
            r = r['predicted']['observed']
            plt.text(3,0,'r = ' + "%.2f" % round(r,2))
            
            #
            #thresholded coupling
            bins = list(np.linspace(-2.5,2.5,40))
            plt.figure()
            coupling = values2['observed'] - values2['predicted']
            n, bins, patches = plt.hist(coupling,bins, normed = 1)
            plt.title('Mutations ' + submodule1 + ' vs. ' + submodule2)
            (mu, sigma) = norm.fit(coupling)
            y = mlab.normpdf( bins, mu, sigma)
            l = plt.plot(bins, y, 'r--', linewidth=2)
            plt.xlim([-2.5,2.5])
            plt.ylim([0,1.3])
            
            plt.text(1,1,'mean = ' + "%.2f" % round(mu,2))
            plt.text(1,0.9,'std.dev = ' + "%.2f" % round(sigma,2))
#%%
#Plot r vs ddG 
data_filt = double_mutant_ddG_summary[double_mutant_ddG_summary['n_corr'] > 5]
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
#%%
double_mutant_ddG_summary.to_pickle('/Users/Steve/Desktop/Data_analysis_code/'
                                          'TL_TLR_CHIP_Data/submodule_coupling_11ntR/'
                                          'double_mutant_ddG_summary_pred_3.pkl') 

#%%
#Effect of mutations to the closing WC base pair can be considered independent 
#of the rest of the motif and can be used to 'tune' the stability of the motif.

#FIND alternative WC base pairs' THIS will only consist of single and double
#mutants otherwise you would have mutations sowewhere else too. 
mask= (single_mutant_ddG_summary['mutations_ 5'] == 1) | (single_mutant_ddG_summary['mutations_ 7'] == 1)
WC_sM = single_mutant_ddG_summary[mask]
mask = (double_mutant_ddG_summary['mutations_ 5'] == 1) & (double_mutant_ddG_summary['mutations_ 7'] == 1)
WC_dM = double_mutant_ddG_summary[mask]

WC_mutants = list(WC_sM.index) + list(WC_dM.index)

WC_effects = pd.DataFrame(index = WC_mutants)
ddG_30Mg = []
ddG_5Mg = []
ddG_5Mg150K = []
for sequence in WC_effects.index:
    if sequence in single_mutant_ddG_summary.index:
        ddG_30Mg.append(single_mutant_ddG_summary.loc[sequence]['ddG_30Mg'])
        ddG_5Mg.append(single_mutant_ddG_summary.loc[sequence]['ddG_5Mg'])
        ddG_5Mg150K.append(single_mutant_ddG_summary.loc[sequence]['ddG_5Mg150K'])
        
    elif sequence in double_mutant_ddG_summary.index:
        ddG_30Mg.append(double_mutant_ddG_summary.loc[sequence]['ddG_30Mg'])
        ddG_5Mg.append(double_mutant_ddG_summary.loc[sequence]['ddG_5Mg'])
        ddG_5Mg150K.append(double_mutant_ddG_summary.loc[sequence]['ddG_5Mg150K'])

#%%
WC_effects['ddG_30Mg'] = ddG_30Mg 
WC_effects['ddG_5Mg'] = ddG_5Mg
WC_effects['ddG_5Mg150K'] = ddG_5Mg150K     
WC_effects['mean'] = WC_effects.mean(axis = 1)
WC_effects['std'] = WC_effects.std(axis = 1)







 










