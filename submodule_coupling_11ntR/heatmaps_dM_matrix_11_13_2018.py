#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:40:49 2018

@author: Steve
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%%


dM_sequence_matrix = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/'
                             'submodule_coupling_11ntR/dM_sequence_matrix.pkl')

double_mutant_ddG_summary = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/'
                                          'TL_TLR_CHIP_Data/submodule_coupling_11ntR/'
                                          'double_mutant_ddG_summary_pred_3.pkl') 

single_mutant_ddG_summary = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/'
                                          'TL_TLR_CHIP_Data/submodule_coupling_11ntR/'
                                          'single_mutant_ddG_summary.pkl')
#%% Generate heatmap with the observed ddGs at 30 mM Mg wrt WT
size_matrix = 36
ddG_matrix_30Mg = pd.DataFrame(np.nan,index = dM_sequence_matrix.index,\
                               columns = dM_sequence_matrix.columns)
#

for each in single_mutant_ddG_summary.index:
    mask = dM_sequence_matrix == each
    ddG_matrix_30Mg[mask] = single_mutant_ddG_summary.loc[each]['ddG_30Mg']
    
    

for each in double_mutant_ddG_summary.index:
    mask = dM_sequence_matrix == each
    ddG_matrix_30Mg[mask] = double_mutant_ddG_summary.loc[each]['ddG_30Mg']

ddG_matrix_30Mg = ddG_matrix_30Mg[ddG_matrix_30Mg.columns[0:size_matrix]]

ddG_matrix_30Mg = ddG_matrix_30Mg.loc[ddG_matrix_30Mg.index[0:size_matrix]]

sns.heatmap(ddG_matrix_30Mg, vmin = -0.5, vmax = 4.5)


#%% Generate heatmap with the observed ddGs at 5 mM Mg 150mM K wrt WT
size_matrix = 36
ddG_matrix_5Mg150K = pd.DataFrame(np.nan,index = dM_sequence_matrix.index,\
                               columns = dM_sequence_matrix.columns)
#

for each in single_mutant_ddG_summary.index:
    mask = dM_sequence_matrix == each
    ddG_matrix_5Mg150K[mask] = single_mutant_ddG_summary.loc[each]['ddG_5Mg150K']
    
    

for each in double_mutant_ddG_summary.index:
    mask = dM_sequence_matrix == each
    ddG_matrix_5Mg150K[mask] = double_mutant_ddG_summary.loc[each]['ddG_5Mg150K']

ddG_matrix_5Mg150K = ddG_matrix_5Mg150K[ddG_matrix_5Mg150K.columns[0:size_matrix]]

ddG_matrix_5Mg150K = ddG_matrix_5Mg150K.loc[ddG_matrix_5Mg150K.index[0:size_matrix]]

sns.heatmap(ddG_matrix_5Mg150K, vmin = -0.5, vmax = 4.5)

#%% Generate heatmap with the observed ddGs at 5 mM Mg wrt WT
size_matrix = 36
ddG_matrix_5Mg = pd.DataFrame(np.nan,index = dM_sequence_matrix.index,\
                               columns = dM_sequence_matrix.columns)
#

for each in single_mutant_ddG_summary.index:
    mask = dM_sequence_matrix == each
    ddG_matrix_5Mg[mask] = single_mutant_ddG_summary.loc[each]['ddG_5Mg']
    
    

for each in double_mutant_ddG_summary.index:
    mask = dM_sequence_matrix == each
    ddG_matrix_5Mg[mask] = double_mutant_ddG_summary.loc[each]['ddG_5Mg']

ddG_matrix_5Mg = ddG_matrix_5Mg[ddG_matrix_5Mg.columns[0:size_matrix]]

ddG_matrix_5Mg = ddG_matrix_5Mg.loc[ddG_matrix_5Mg.index[0:size_matrix]]

sns.heatmap(ddG_matrix_5Mg, vmin = -0.5, vmax = 4.5)
#%% Generate heatmap for the predicited ddG of double mutants based on the effects
#of the single mutants wrt to WT; that is, additive model
#do this comparison at 30 mM Mg
size_matrix = 36
ddG_matrix_30Mg_pred = pd.DataFrame(np.nan,index = dM_sequence_matrix.index,\
                               columns = dM_sequence_matrix.columns)
#

for each in single_mutant_ddG_summary.index:
    mask = dM_sequence_matrix == each
    ddG_matrix_30Mg_pred[mask] = np.nan
    
for each in double_mutant_ddG_summary.index:
    mask = dM_sequence_matrix == each
    ddG_matrix_30Mg_pred[mask] = double_mutant_ddG_summary.loc[each]['pred_add_30Mg']

ddG_matrix_30Mg_pred = ddG_matrix_30Mg_pred[ddG_matrix_30Mg_pred.columns[0:size_matrix]]

ddG_matrix_30Mg_pred = ddG_matrix_30Mg_pred.loc[ddG_matrix_30Mg_pred.index[0:size_matrix]]

sns.heatmap(ddG_matrix_30Mg_pred, vmin = -0.5, vmax = 4.5)

#%% Generate heatmap for the predicited ddG of double mutants based on the effects
#of the single mutants wrt to WT; that is, additive model
#do this comparison at 5 mM Mg 150K
size_matrix = 36
ddG_matrix_5Mg150K_pred = pd.DataFrame(np.nan,index = dM_sequence_matrix.index,\
                               columns = dM_sequence_matrix.columns)
#

for each in single_mutant_ddG_summary.index:
    mask = dM_sequence_matrix == each
    ddG_matrix_5Mg150K_pred[mask] = np.nan
    
for each in double_mutant_ddG_summary.index:
    mask = dM_sequence_matrix == each
    ddG_matrix_5Mg150K_pred[mask] = double_mutant_ddG_summary.loc[each]['pred_add_5Mg150K']

ddG_matrix_5Mg150K_pred = ddG_matrix_5Mg150K_pred[ddG_matrix_5Mg150K_pred.columns[0:size_matrix]]

ddG_matrix_5Mg150K_pred = ddG_matrix_5Mg150K_pred.loc[ddG_matrix_5Mg150K_pred.index[0:size_matrix]]

sns.heatmap(ddG_matrix_5Mg150K_pred, vmin = -0.5, vmax = 4.5)


#%% Generate heatmap for the predicited ddG of double mutants based on the effects
#of the single mutants wrt to WT; that is, additive model
#do this comparison at 5 mM Mg 
size_matrix = 36
ddG_matrix_5Mg_pred = pd.DataFrame(np.nan,index = dM_sequence_matrix.index,\
                               columns = dM_sequence_matrix.columns)
#

for each in single_mutant_ddG_summary.index:
    mask = dM_sequence_matrix == each
    ddG_matrix_5Mg_pred[mask] = np.nan
    
for each in double_mutant_ddG_summary.index:
    mask = dM_sequence_matrix == each
    ddG_matrix_5Mg_pred[mask] = double_mutant_ddG_summary.loc[each]['pred_add_5Mg']

ddG_matrix_5Mg_pred = ddG_matrix_5Mg_pred[ddG_matrix_5Mg_pred.columns[0:size_matrix]]

ddG_matrix_5Mg_pred = ddG_matrix_5Mg_pred.loc[ddG_matrix_5Mg_pred.index[0:size_matrix]]

sns.heatmap(ddG_matrix_5Mg_pred, vmin = -0.5, vmax = 4.5)
#%% Heatmap of coupling (double mutant observed vs predicted from additive model)
# at 30 mM Mg
coupling_30Mg = ddG_matrix_30Mg - ddG_matrix_30Mg_pred
coupling_30Mg_heatmap = sns.heatmap(coupling_30Mg,cmap = 'coolwarm', vmin = -4, vmax = 4)
fig = coupling_30Mg_heatmap.get_figure()
fig.savefig('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/submodule_coupling_11ntR/coupling_30Mg_heatmap.svg')

#%% Heatmap of coupling (double mutant observed vs predicted from additive model)
# at 5 mM Mg + 150 K 
coupling_5Mg150K = ddG_matrix_5Mg150K - ddG_matrix_5Mg150K_pred
coupling_5Mg150K_heatmap = sns.heatmap(coupling_5Mg150K,cmap = 'coolwarm', vmin = -4, vmax = 4)
fig = coupling_5Mg150K_heatmap.get_figure()
fig.savefig('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/submodule_coupling_11ntR/coupling_5Mg150K_heatmap.svg')
#%% Heatmap of coupling (double mutant observed vs predicted from additive model)
# at 5 mM Mg 
coupling_5Mg = ddG_matrix_5Mg - ddG_matrix_5Mg_pred
coupling_5Mg_heatmap = sns.heatmap(coupling_5Mg,cmap = 'coolwarm', vmin = -4, vmax = 4)
fig = coupling_5Mg_heatmap.get_figure()
fig.savefig('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/submodule_coupling_11ntR/coupling_5Mg_heatmap.svg')
#%%
#coupling with vs without potassium in 5mM Mg
potassium_coupling = coupling_5Mg150K - coupling_5Mg
potassium_coupling_heatmap = sns.heatmap(potassium_coupling,cmap = 'coolwarm', vmin = -4, vmax = 4)
fig = potassium_coupling_heatmap.get_figure()
fig.savefig('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/submodule_coupling_11ntR/potassium_coupling_heatmap.svg')
#%%
#coupling at 30 vs coupling at 5 mM Mg 
magnesium_coupling = coupling_30Mg - coupling_5Mg
magnesium_coupling_heatmap = sns.heatmap(magnesium_coupling,cmap = 'coolwarm', vmin = -4, vmax = 4)
fig = magnesium_coupling_heatmap.get_figure()
fig.savefig('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/submodule_coupling_11ntR/magnesium_coupling_heatmap.svg')




#%%
A = coupling_30Mg.copy()
A[(A>4) | (A<-4)] = np.nan
B = coupling_5Mg.copy()
B[(B>4) | (B<-4)] = np.nan
C = B - A
sns.heatmap(C,cmap = 'coolwarm', vmin = -4, vmax = 4)


#%%
x = double_mutant_ddG_summary['pred_add_30Mg']
y = double_mutant_ddG_summary['pred_add_5Mg']
plt.scatter(x,y)
plt.plot(x,x,'--')

plt.figure()
x = double_mutant_ddG_summary['ddG_30Mg']
y = double_mutant_ddG_summary['ddG_5Mg']
plt.scatter(x,y)
plt.plot(x,x,'--')
#%%
#mutation to WC and platform 
mask = (double_mutant_ddG_summary['WC_mutations'])  & (double_mutant_ddG_summary['platform_mutations'])
data = double_mutant_ddG_summary[mask]
data_seqs = list(data.index)
   