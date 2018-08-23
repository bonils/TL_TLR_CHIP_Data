#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:35:33 2018

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
from scipy import stats
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
high_lim = -5
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
#%%
'''---------------Import data ------------------'''
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
all_data = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
all_data = all_data.drop_duplicates(subset='seq')
all_data['new_name'] = all_data['r_name'] + '_' + all_data['r_seq'] + '_' +  all_data['sublibrary']
all_data = all_data[all_data.b_name == 'normal']
receptor_group = all_data.groupby(by='new_name')
dG_avg_30mM_GAAA = receptor_group['dG_Mut2_GAAA'].agg(['mean','count'])
dG_avg_30mM_GAAA_50scaff = dG_avg_30mM_GAAA[dG_avg_30mM_GAAA['count'] > 40]
#HOW MANY OF THE MOST STABLE
n_stable = 200
most_stable = dG_avg_30mM_GAAA_50scaff.nsmallest(n_stable,'mean')
r_values = pd.DataFrame(index=most_stable.index)
r_values['mean_dG_30mM'] = most_stable['mean']
r_values_list = []

#values above certain threshold are replaced by nan
to_nan = 1
new_threshold = -7.1#dG_threshold
#values below certain n threshold are plotted
apply_n_thr = 1
n_thr = 3

n_total_list = []
n_compare_list = []

for receptors in most_stable.index:
    x_data = receptor_group.get_group(receptors)['dG_Mut2_GAAA'].copy()
    y_data = receptor_group.get_group(receptors)['dG_Mut2_GAAA_5mM_2'].copy()
    if to_nan == 1:
        x_data[x_data>new_threshold] = np.nan
        y_data[y_data>new_threshold] = np.nan
    n_total = len(x_data)
    n_total_list.append(n_total)
    n_compare = sum(~(x_data.isna() | y_data.isna()))
    n_compare_list.append(n_compare)
    if apply_n_thr == 1:
        if n_compare > n_thr:
            r_values_list.append(x_data.corr(y_data))
        else:
            r_values_list.append(np.nan)
    else:
        r_values_list.append(x_data.corr(y_data))
r_values['r_pearson_all_data'] = r_values_list
r_values['n_total'] = n_total_list
r_values['n_compare'] = n_compare_list

plt.scatter(r_values['mean_dG_30mM'],r_values['r_pearson_all_data'])
plt.ylim([-1,1])
#y_data = pd.DataFrame(y_data).dropna()
#df = x_data.merge(y_data,left_index=True, right_index=True, how='left')
#x_on_y = pd.ols(y=df[df.columns[0]], x=df[df.columns[1]], intercept=True)
#print(df[df.columns[0]].corr(df[df.columns[1]]), x_on_y.f_stat['p-value'])
#%%
r, p = stats.pearsonr(x_data.dropna(),y_data.dropna())
print(r)
print(p)
#%%






all_11ntR = pd.read_csv(data_path + 'all_11ntRs_unique.csv' )
all_11ntR_normal = all_11ntR[all_11ntR.b_name == 'normal']
unique_11ntR_receptors = list(set(all_11ntR_normal.r_seq))
#Calculate average dG for each of them
receptor_groups = all_11ntR_normal.groupby(['r_seq'])
dG_30 = receptor_groups['dG_Mut2_GAAA']
dG_5 = receptor_groups['dG_Mut2_GAAA_5mM_2']

#HOW MANY OF THE MOST STABLE
n_stable = 15

dG_avg_30mM= all_11ntR_normal.groupby(['r_seq'])['dG_Mut2_GAAA'].agg(['mean', 'count'])
dG_avg_30mM_50scaff = dG_avg_30mM[dG_avg_30mM['count'] > 40]
most_stable = dG_avg_30mM_50scaff.nsmallest(n_stable,'mean')
for receptors in most_stable.index:
    plt.figure()
    plt.scatter(dG_30.get_group(receptors),dG_5.get_group(receptors),s=120,edgecolors='k')
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
    
    
    x_data = dG_30.get_group(receptors)
    y_data = dG_5.get_group(receptors)
    with_data = ~((x_data.isna()) | (y_data.isna()))
    data_points_total = with_data.sum() 
#    x_data[x_data > dG_threshold] = np.nan
#    y_data[y_data > dG_threshold] = np.nan
    ddG = y_data.subtract(x_data)
    above_limit = ~ddG.isnull()
    n_above_limit = above_limit.sum()
    ddG_avg = ddG.mean()
    ddG_std = ddG.std()
    r_pearson = x_data.corr(y_data)
    r_sq = r_pearson**2
    textstr = 'n=%.0f\nn$_{thr}$ =%.0f\n$\Delta\Delta G=%.2f$\n$\mathrm{r}=%.2f$' % (data_points_total,n_above_limit,ddG_avg, r_sq)
    plt.text(-9, -10, textstr, fontsize=7,
    verticalalignment='top')
    x_ddG = [ddG_avg + x[0],ddG_avg + x[1]]
    plt.plot(x,x_ddG,'--r',linewidth = 3)
    
    
    
    
    
