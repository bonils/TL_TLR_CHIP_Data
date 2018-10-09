#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 18:01:07 2018

@author: Steve
"""

#%%
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
from sklearn.metrics import mean_squared_error
from math import sqrt
import random
#%%
'''--------------Import Functions--------------''' 
from clustering_functions import get_dG_for_scaffold
from clustering_functions import doPCA
from clustering_functions import interpolate_mat_knn
from clustering_functions import prep_data_for_clustering_ver2
from clustering_functions import prep_data_for_clustering_ver3


'''---------------General Variables-------------'''
dG_threshold = -7.1 #kcal/mol; dG values above this are not reliable
dG_replace = -7.1 # for replacing values above threshold. 
nan_threshold = 0.50 #amount of missing data tolerated.
num_neighbors = 10
#for plotting
low_lim = -14
high_lim = -5

'''---------------Import data ------------------'''
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
entire_lib = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')

entire_lib = entire_lib.drop_duplicates(subset ='seq')
#Consider the ones only with normal closing base pair 
mask = entire_lib.b_name == 'normal' 
entire_lib_normal_bp = entire_lib[mask]
#Exclude sublibrary 5 which has the mutation intermediates and cannot be easily classified (or maybe they can be classified as others)
mask = (entire_lib_normal_bp.sublibrary == 'tertcontacts_0') | (entire_lib_normal_bp.sublibrary == 'tertcontacts_1')|\
       (entire_lib_normal_bp.sublibrary == 'tertcontacts_2') | (entire_lib_normal_bp.sublibrary == 'tertcontacts_3')|\
       (entire_lib_normal_bp.sublibrary == 'tertcontacts_4') | (entire_lib_normal_bp.sublibrary == 'tertcontacts_5')
entire_lib_selected = entire_lib_normal_bp[mask].copy()
#%Create new name identifier because there are some that were given the same name
#but they are different receptors
#also attacht the sublibrary they came from for later reference
entire_lib_selected['new_name'] = entire_lib_selected.r_name + '_' + entire_lib_selected.r_seq #+ '_' + entire_lib_selected.sublibrary
#entire_lib_selected = entire_lib_selected.set_index('new_name')
grouped_lib = entire_lib_selected.groupby('r_seq')
WT_data = grouped_lib.get_group('UAUGG_CCUAAG')
WT_data = WT_data.set_index('old_idx')

B = WT_data
C = B.groupby('length')
s = []
for length in C.groups:
    next_group = C.get_group(length)
    next_group = next_group.sort_values('dG_Mut2_GAAA',ascending = False)
    s.append(list(next_group.index))

sc = [item for sublist in s for item in sublist]
B = B.reindex(sc)
sc8 = list(B[B['length'] == 8].index)
sc9 = list(B[B['length'] == 9].index)
sc10 = list(B[B['length'] == 10].index)
sc11= list(B[B['length'] == 11].index)
#
#B = B.sort_values('length')
fifty_scaffolds_bylength = B.index

#%%
data_to_plot = pd.DataFrame(index = B.index)
data_to_plot['wt'] = WT_data['dG_Mut2_GAAA']

same = 'UAUGG_CCUUAG'
same_data = grouped_lib.get_group(same)
same_data = same_data.set_index('old_idx')
same_data = same_data.reindex(fifty_scaffolds_bylength)
data_to_plot['same'] = same_data['dG_Mut2_GAAA']

different = 'GGAGG_CCUAAAC'
#different = 'UGUGG_CCUAAG'
different_data = grouped_lib.get_group(different)
different_data = different_data.set_index('old_idx')
different_data = different_data.reindex(fifty_scaffolds_bylength)
data_to_plot['different'] = different_data['dG_Mut2_GUAA_1']

data_to_plot = data_to_plot.dropna()

plt.figure()
plt.bar(range(len(data_to_plot)),data_to_plot['wt'] * -1)#,edgecolor = 'black')
plt.ylim(5,14)
plt.plot(range(len(data_to_plot)),data_to_plot['wt'] * -1,linewidth = 3,color = 'black')

plt.figure()
plt.bar(range(len(data_to_plot)),data_to_plot['same'] * -1,color = 'green')#,edgecolor = 'black')
plt.ylim(5,14)
plt.plot(range(len(data_to_plot)),data_to_plot['same'] * -1,linewidth = 3,color = 'black')

plt.figure()
plt.bar(range(len(data_to_plot)),data_to_plot['different'] * -1,color = 'purple')#,edgecolor = 'black')
plt.ylim(5,14)
plt.plot(range(len(data_to_plot)),data_to_plot['different'] * -1,linewidth = 3,color = 'black')

ddG_average = data_to_plot['same'].subtract(data_to_plot['wt']).median()

color_a = 'green'
plt.figure()
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
x_ddG = [ddG_average + x[0],ddG_average + x[1]]
plt.plot(x,x_ddG,'--r',linewidth = 3)
plt.scatter(data_to_plot['wt'],data_to_plot['same'],s=120,edgecolors='k',c=color_a,linewidth=2)
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


ddG_average = data_to_plot['different'].subtract(data_to_plot['wt']).median()

color_a = 'purple'
plt.figure()
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
x_ddG = [ddG_average + x[0],ddG_average + x[1]]
plt.plot(x,x_ddG,'--r',linewidth = 3)
plt.scatter(data_to_plot['wt'],data_to_plot['different'],s=120,edgecolors='k',c=color_a,linewidth=2)
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
#%%
variant1 = 'UAUGG_CCUAAA'
variant1_data = grouped_lib.get_group(variant1)
variant1_data = variant1_data.set_index('old_idx')
variant1_data = variant1_data.reindex(fifty_scaffolds_bylength)
#variant1_data = variant1_data['dG_Mut2_GAAA'].sub(variant1_data['dG_Mut2_GAAA'].median())
#plt.bar(range(50),variant1_data)
plt.figure()
plt.bar(range(50),variant1_data['dG_Mut2_GAAA'] * -1)#,edgecolor = 'black')
plt.ylim(6,12)
plt.plot(range(50),variant1_data['dG_Mut2_GAAA'] * -1,linewidth = 3,color = 'black')
#%%
variant2 = 'UAUGG_CCUACG'
variant2_data = grouped_lib.get_group(variant2)
variant2_data = variant2_data.set_index('old_idx')
variant2_data = variant2_data.reindex(fifty_scaffolds_bylength)
#variant1_data = variant1_data['dG_Mut2_GAAA'].sub(variant1_data['dG_Mut2_GAAA'].median())
#plt.bar(range(50),variant1_data)
plt.figure()
plt.bar(range(50),variant2_data['dG_Mut2_GAAA'] * -1)#,edgecolor = 'black')
plt.ylim(6,12)
plt.plot(range(50),variant2_data['dG_Mut2_GAAA'] * -1,linewidth = 3,color = 'black')
#%%
variant1 = 'UAUGG_CCUAAA'
variant1_data = grouped_lib.get_group(variant1)
variant1_data = variant1_data.set_index('old_idx')
variant1_data = variant1_data.reindex(fifty_scaffolds_bylength)
variant1_data = variant1_data['dG_Mut2_GUAA_1']

plt.bar(range(50),variant1_data['dG_Mut2_GUAA_1'] * -1)#,edgecolor = 'black')

plt.ylim(6,12)
plt.plot(range(50),variant1_data['dG_Mut2_GUAA_1'] * -1,linewidth = 3,color = 'black')
#%%
A = pd.DataFrame()
A['GUAA'] = grouped_lib['dG_Mut2_GUAA_1'].median()
A['n'] = grouped_lib['dG_Mut2_GUAA_1'].count()
A = A[A['GUAA'] < -8]



