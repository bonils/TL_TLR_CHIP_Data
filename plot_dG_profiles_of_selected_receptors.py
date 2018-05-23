#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:47:46 2018

@author: Steve
"""
'''Sequence to plot wrt to 11ntR'''
alt_TLR_seq = 'CACGG_CCUAGA'

'''Condition to compare'''
condition = 'dG_Mut2_GAAA'  # for 30 mM Mg

'''Limits for plot figures'''
low_lim = -14
high_lim = -6

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
#%%
'''Calculate ddG values'''
ddG = alt_TLR[condition] - WT_11ntR[condition] 
ddG_average = ddG.mean()
ddG_std = ddG.std()

#%%
#Color based on the length of the CHIP piece 
Colors = WT_11ntR.length.copy()
Colors[Colors == 8] = 'red'
Colors[Colors == 9] = 'blue'
Colors[Colors == 10] = 'orange'
Colors[Colors == 11] = 'black'
#%%
fig1 = plt.figure()
x = [low_lim, high_lim] #for plotting  y= x line
plt.scatter(WT_11ntR[condition],alt_TLR[condition],s=60,edgecolors='k',c=Colors)
plt.plot(x,x,'--r')
plt.xlim(low_lim,high_lim)
plt.ylim(low_lim,high_lim)
plt.xticks(list(range(-14,-4,2)))
plt.yticks(list(range(-14,-4,2)))
plt.tick_params(axis='both', which='major', labelsize=20)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta$G$^{WT}_{bind}$ (kcal/mol)',fontsize=18)
plt.ylabel('$\Delta$G$^{alt}_{bind}$ (kcal/mol)',fontsize=18)
fig1.show()