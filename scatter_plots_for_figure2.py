#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:27:13 2018

@author: Steve
"""

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
from sklearn.preprocessing import StandardScaler
#%%
'''--------------Import Functions--------------''' 
from clustering_functions import get_dG_for_scaffold
from clustering_functions import doPCA
from clustering_functions import interpolate_mat_knn
from clustering_functions import prep_data_for_clustering_ver2
from clustering_functions import prep_data_for_clustering_ver3
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

#%%
sublib0 = entire_lib[entire_lib['sublibrary'] =='tertcontacts_0']
sublib0_group = sublib0.groupby('r_name')
#%%
WT_data = sublib0_group.get_group('11ntR')
WT_data = WT_data.set_index('old_idx')
#%%
###plot
low_lim = -14
high_lim = -6
receptor = 'C7.2'
color = 'orange'

mutant_data = sublib0_group.get_group(receptor)
mutant_data = mutant_data.set_index('old_idx')
mutant_data = mutant_data.reindex(WT_data.index)
#%%
plt.figure()

#plot 30 mM GAAA
plt.scatter(WT_data['dG_Mut2_GAAA'],mutant_data['dG_Mut2_GAAA'],s=200,edgecolors='k',c=color)

#plot 5 mM GAAA
plt.scatter(WT_data['dG_Mut2_GAAA_5mM_2'],mutant_data['dG_Mut2_GAAA_5mM_2'],s=200,edgecolors='k',c=color,marker='^')

#plot 5 mM + 150K
plt.scatter(WT_data['dG_Mut2_GAAA_5mM_150mMK_1'],mutant_data['dG_Mut2_GAAA_5mM_150mMK_1'],s=200,edgecolors='k',c=color,marker ='*')

#plot 30mM GUAA
plt.scatter(WT_data['dG_Mut2_GUAA_1'],mutant_data['dG_Mut2_GUAA_1'],s=200,edgecolors='k',c=color,marker ='s')

#%%
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













