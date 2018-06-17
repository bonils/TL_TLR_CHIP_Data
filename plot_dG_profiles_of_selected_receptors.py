#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:47:46 2018

@author: Steve
"""
'''Sequence to plot wrt to 11ntR'''
alt_TLR_seq = 'CAUGG_CCUAAG'

'''Condition to compare'''
condition = 'dG_Mut2_GAAA'  # for 30 mM Mg
error = 'dGerr_Mut2_GAAA'

'''Limits for plot figures'''
low_lim = -14
high_lim = -6

'''Set threshold'''
dG_threshold = -7.1
set_threshold = True

'''plot originial or thresholded data'''
plot_original = True
#%%
'''Import libraries'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
'''Import Data'''
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
data_11ntR = pd.read_csv(data_path + 'all_11ntRs_unique.csv')
print('original size of dataframe with all 11ntR is:',data_11ntR.shape)
#Select data only with normal flanking base pairs
data_11ntR = data_11ntR[data_11ntR.b_name == 'normal']
print('data with normal flaking base pairs is: ', data_11ntR.shape)
#%%
'''WT'''
# create dataframe with WT receptors and flanking bp == 'normal'
WT_11ntR = data_11ntR[data_11ntR.r_seq == 'UAUGG_CCUAAG']
print('size of WT dataframe:',WT_11ntR.shape)
#%%
unique_scaffolds = set(WT_11ntR['old_idx'])
WT_11ntR = WT_11ntR.set_index('old_idx')
WT_11ntR = WT_11ntR.reindex(unique_scaffolds)
#%%
alt_TLR = data_11ntR[data_11ntR.r_seq == alt_TLR_seq]
alt_TLR = alt_TLR.set_index('old_idx')
alt_TLR = alt_TLR.reindex(unique_scaffolds)

#%% Keep original data without thresholding for plotting.

WT_11ntR_original = WT_11ntR.copy()
alt_TLR_original = alt_TLR.copy()
#%%
'''Set values above dG threshold to NAN'''
if set_threshold:
    cond = alt_TLR[condition].copy()
    cond[cond>dG_threshold] = np.nan
    alt_TLR[condition] = cond
    del cond
    cond = WT_11ntR[condition].copy()
    cond[cond>dG_threshold] = np.nan
    WT_11ntR[condition] = cond
#%%
#ddG and correlation values are calculated based on thresholfed data
'''Calculate ddG values'''
ddG = alt_TLR[condition] - WT_11ntR[condition] 
ddG_average = ddG.mean()
ddG_std = ddG.std()

'''Correlation coefficient of thresholded data'''
data_comb = pd.concat([alt_TLR[condition],WT_11ntR[condition]],axis = 1)
R = data_comb.corr(method = 'pearson')

'''Correlation coefficient original data'''
data_comb_original = pd.concat([alt_TLR_original[condition],WT_11ntR_original[condition]],axis = 1)
R_original = data_comb_original.corr(method = 'pearson')
#%%
#Color based on the length of the CHIP piece 
Colors = WT_11ntR.length.copy()
Colors[Colors == 8] = 'red'
Colors[Colors == 9] = 'blue'
Colors[Colors == 10] = 'orange'
Colors[Colors == 11] = 'black'

#%%
if plot_original:
   fig1 = plt.figure()
   x = [low_lim, dG_threshold] #for plotting  y= x line
   y_thres = [dG_threshold,dG_threshold]
   x_ddG = [ddG_average + x[0],ddG_average + x[1]]
   plt.plot(x,x_ddG,'--r',linewidth = 3)
   plt.scatter(WT_11ntR_original[condition],alt_TLR_original[condition],s=120,edgecolors='k',c=Colors)
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
   fig1.show() 
else:
    fig1 = plt.figure()
    x = [low_lim, dG_threshold] #for plotting  y= x line
    y_thres = [dG_threshold,dG_threshold]
    x_ddG = [ddG_average + x[0],ddG_average + x[1]]
    plt.plot(x,x_ddG,'--r',linewidth = 3)
    plt.scatter(WT_11ntR[condition],alt_TLR[condition],s=120,edgecolors='k',c=Colors)
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
    fig1.show()

fig1.savefig(alt_TLR_seq + '.svg')
#%%
