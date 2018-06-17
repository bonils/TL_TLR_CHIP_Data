#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 21:19:50 2018

@author: Steve
"""
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
num_neighbors = 10
#for plotting
low_lim = -14
high_lim = -6
#%%
'''---------------Import data ------------------'''
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
tecto_data = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
#%%
alt_name = 'UCUGG_CCUAAG'
condition = 'dG_Mut2_GAAA'

alt_mask = tecto_data.r_seq == alt_name
alt_data = tecto_data[alt_mask]

WT_mask = (tecto_data['r_seq'] == 'UAUGG_CCUAAG') & (tecto_data['sublibrary'] == 'tertcontacts_0')
WT_data = tecto_data[WT_mask]


scaffolds = list(set(WT_data['old_idx']))


WT_data = WT_data.set_index('old_idx')
alt_data = alt_data.set_index('old_idx')

WT_data = WT_data.reindex(scaffolds)

alt_data = alt_data.drop_duplicates(subset='seq', keep="last")
alt_data = alt_data.reindex(scaffolds)

x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]

#Calculate R_sq and ddG
alt_data_thrs = alt_data[condition].copy()
WT_data_thrs = WT_data[condition].copy()
alt_data_thrs[alt_data_thrs>dG_threshold] = np.nan
WT_data_thrs[WT_data_thrs>dG_threshold] = np.nan


#x_ddG = [ddG_average + x[0],ddG_average + x[1]]
#plt.plot(x,x_ddG,'--r',linewidth = 3)
plt.scatter(WT_data.dG_Mut2_GAAA,alt_data.dG_Mut2_GAAA,s=120,edgecolors='k',marker='o')
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
