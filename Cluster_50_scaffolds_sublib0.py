#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 13:07:41 2018

@author: Steve
"""
# Import sublibrary 0 and create a table with rows being each of the ~150 tetraloop
# receptors and each of the columns being dG measurements.
# Because there were 50 scaffolds for each TLR and each of these was measured
# under 3 different ionic conditions and an alternative GUUA TL, there are
# 200 columns. 

# Each TLR was 'normalized' wrt its mean across columns. 

# PCA, and determine how many PCs to used for hierarchical clustering. 
# Hierarchical clustering, using ward distance, and determine the cutoff distance.

#Symbols used in scatter plots:
#o --> 30 mM Mg, GAAA
#s --> 30 mM Mg, GUAA
#^ --> 5 mM Mg
#


#%%
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
sublib0 = tecto_data[tecto_data['sublibrary'] == 'tertcontacts_0']
print('original size of sublibrary 0: ',sublib0.shape)
sublib0 = sublib0.drop_duplicates(subset='seq', keep="last")
print('after deleting duplicates: ',sublib0.shape)
#%%
#Create list of receptors in sublibrary and list of scaffolds
receptors_sublib0 = sorted(list(set(sublib0['r_name'])))
scaffolds_length = pd.concat([sublib0['old_idx'],sublib0['length']],axis=1)
scaffolds_length = scaffolds_length.drop_duplicates()
scaffolds_length = scaffolds_length.sort_values(by=['length'])
scaffolds_sublib0 = list(scaffolds_length['old_idx'])
#%% Colors to used for scatter plotting
Color_length =scaffolds_length['length'].copy()
Color_length[Color_length == 8] = 'red'
Color_length[Color_length == 9] = 'blue'
Color_length[Color_length == 10] = 'orange'
Color_length[Color_length == 11] = 'black'
#%%
#Note: in previous datatables (e.g. all_11ntR_unique.csv) old_idx imports as a 
# number.  However in the case of the original table 'tectorna_results_tertcontacts.180122.csv'
# old_idx imports as a string.
data = sublib0
conditions = ['dG_Mut2_GAAA']#, 'dG_Mut2_GUAA_1']
column_labels = ['dG_30mM_Mg_GAAA']#,'dG_30mM_Mg_GUAA']
row_index= ('r_name')
flanking = 'normal'

data_50_scaffolds_GAAA = pd.DataFrame(index = receptors_sublib0)

for scaffolds in scaffolds_sublib0:
    
    next_scaffold = str(scaffolds)
    next_column_labels = [name + '_' +  next_scaffold for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(receptors_sublib0)
    data_50_scaffolds_GAAA = pd.concat([data_50_scaffolds_GAAA,next_df],axis = 1)


conditions = ['dG_Mut2_GUAA_1']
column_labels = ['dG_30mM_Mg_GUAA']
row_index= ('r_name')
flanking = 'normal'

data_50_scaffolds_GUAA = pd.DataFrame(index = receptors_sublib0)

for scaffolds in scaffolds_sublib0:
    
    next_scaffold = str(scaffolds)
    next_column_labels = [name + '_' +  next_scaffold for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(receptors_sublib0)
    data_50_scaffolds_GUAA = pd.concat([data_50_scaffolds_GUAA,next_df],axis = 1)


conditions = ['dG_Mut2_GAAA_5mM_2']
column_labels = ['dG_5mM_Mg_GAAA']
row_index= ('r_name')
flanking = 'normal'

data_50_scaffolds_5Mg = pd.DataFrame(index = receptors_sublib0)

for scaffolds in scaffolds_sublib0:
    
    next_scaffold = str(scaffolds)
    next_column_labels = [name + '_' +  next_scaffold for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(receptors_sublib0)
    data_50_scaffolds_5Mg = pd.concat([data_50_scaffolds_5Mg,next_df],axis = 1)


conditions = ['dG_Mut2_GAAA_5mM_150mMK_1']
column_labels = ['dG_5Mg150K_GAAA']
row_index= ('r_name')
flanking = 'normal'

data_50_scaffolds_5Mg150K = pd.DataFrame(index = receptors_sublib0)

for scaffolds in scaffolds_sublib0:
    
    next_scaffold = str(scaffolds)
    next_column_labels = [name + '_' +  next_scaffold for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(receptors_sublib0)
    data_50_scaffolds_5Mg150K = pd.concat([data_50_scaffolds_5Mg150K,next_df],axis = 1)
    
data_50_scaffolds = pd.concat([data_50_scaffolds_GAAA,data_50_scaffolds_GUAA,
                               data_50_scaffolds_5Mg,
                               data_50_scaffolds_5Mg150K],axis = 1)
    
prep_data,original_nan = prep_data_for_clustering_ver2(data_50_scaffolds,
                                                       dG_threshold,dG_replace,nan_threshold,
                                                       num_neighbors=num_neighbors)
prep_data_with_nan = prep_data.copy()
prep_data_with_nan[original_nan] = np.nan
#%%
#Normalized each tetraloop-receptor with respect to its mean accross columns
mean_per_row = prep_data.mean(axis=1)
prep_data_norm = prep_data.copy()
prep_data_norm = prep_data_norm.sub(mean_per_row,axis=0)
prep_data_norm_with_nan = prep_data_norm.copy()
prep_data_norm_with_nan[original_nan] = np.nan
#%%
'''------------PCA Analysis of raw data-----------------'''
pca,transformed,loadings = doPCA(prep_data_norm)
#plot explained variance by PC
pd.Series(pca.explained_variance_ratio_).plot(kind='bar')
plt.ylabel('fraction of variance \nexplained by each PC', fontsize=14)
plt.tight_layout()
num_PCA = 12
print('Fraction explained by the first ',str(num_PCA), 'PCAs :',sum(pca.explained_variance_ratio_[:num_PCA]))
#%%
list_PCAs = list(transformed.columns[:num_PCA])
z_pca = sch.linkage(transformed.loc[:,list_PCAs],method='ward') 
cg_pca = sns.clustermap(prep_data_norm_with_nan,row_linkage=z_pca, col_cluster=False
                        , vmin=-3,vmax=3,cmap='coolwarm')
X = transformed.loc[:,list_PCAs]
c, coph_dists = cophenet(z_pca, pdist(X))
print('cophenetic distance: ',c)
plt.show()
#%%
distance_threshold = 15
sch.dendrogram(z_pca,color_threshold=distance_threshold)
max_d = distance_threshold
clusters = fcluster(z_pca, max_d, criterion='distance')
number_clusters = max(clusters)
print('number of clusters based on distance of ',str(max_d), ':', str(number_clusters))
# Append cluster number to dataframe with data
clustered_data = prep_data.copy()
clustered_data_nan = prep_data_with_nan.copy()
cluster_series = pd.Series(clusters,index=clustered_data.index)
clustered_data['cluster'] = cluster_series
clustered_data_nan['cluster'] = cluster_series

#Replot clustergram with rows colored as above
row_color = pd.Series(clusters,index=clustered_data.index)
cluster_colors = ['g','r','c','m','y','k','b','orange','g','r','c','m','y','k','b']
num_clusters = len(cluster_series.unique())
num_clusters
for i in range(num_clusters):
    row_color[row_color == (i+1)] = cluster_colors[i]
 
cg_pca_col = sns.clustermap(prep_data_norm_with_nan,row_linkage=z_pca,
                            col_cluster=False, vmin=-4,vmax=4,row_colors=row_color,
                            cmap='coolwarm')

#Append type and sequence of TLRs
receptors = clustered_data.index
types = []
for names in receptors:        
    if 'C7' in names or 'B7' in names:
        types.append(2)
    elif 'IC3' in names or 'VC2' in names:
        types.append(3)
    elif '11ntR' in names:
        types.append(1)
    elif 'tandem' in names:
        types.append(4) 
    else:
        types.append(5)
types_series = pd.Series(types,index = clustered_data.index)
clustered_data['receptor_type'] = types_series
clustered_data_nan['receptor_type'] = types_series

#separate each cluster and save in dictionary
cluster_names = ['cluster_' + str(i) for i in range(1,number_clusters+1)]
all_clusters = {}
all_clusters_nan = {}
for counter, names in enumerate (cluster_names):
    all_clusters[names] = clustered_data.query('cluster==' + str(counter +1))
    all_clusters_nan[names] = clustered_data_nan.query('cluster==' + str(counter +1))
#how many members are in each cluster
for clusters in all_clusters:
    s1,s2 = all_clusters[clusters].shape
    print('There are ',str(s1),' tetraloop-receptors in ', clusters)
#%%
'''----------------Plot a random member of cluster n-------------------'''    
cluster_to_plot = 'cluster_10'
WT_ref = data_50_scaffolds.loc['11ntR'] 
next_cluster = all_clusters_nan[cluster_to_plot]
S1,S2 = next_cluster.shape
rand_index = np.random.randint(1,S1)
alternative_TLR = next_cluster.index[rand_index]
alt_TLR = data_50_scaffolds.loc[alternative_TLR]
r_pearson = alt_TLR.corr(WT_ref)

x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
#x_ddG = [ddG_average + x[0],ddG_average + x[1]]
#plt.plot(x,x_ddG,'--r',linewidth = 3)
plt.scatter(WT_ref[0:50],alt_TLR[0:50],s=120,edgecolors='k',c=Color_length,marker='o')
plt.scatter(WT_ref[50:100],alt_TLR[50:100],s=120,edgecolors='k',c=Color_length,marker='s')
plt.scatter(WT_ref[100:150],alt_TLR[100:150],s=120,edgecolors='k',c=Color_length,marker='^')
plt.scatter(WT_ref[150:200],alt_TLR[150:200],s=120,edgecolors='k',c=Color_length,marker='*')
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
plt.title(alt_TLR.name)
#plt.savefig(cluster_to_plot + '_fig.svg')
#%%%
'''-----------Plot all members of cluster n relative to 11ntR---------------'''
cluster_to_plot = 'cluster_1'
panels_per_figure = 4

data_cluster = all_clusters_nan[cluster_to_plot]

WT_ref = data_50_scaffolds.loc['11ntR']
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
s1,s2 = data_cluster.shape
if s1%panels_per_figure > 0:
    num_figs = int(s1/panels_per_figure) + 1
else:
    num_figs = int(s1/panels_per_figure)
axes = []

for i in range(num_figs):
    fig1,axes1 = plt.subplots(2,2,sharex=True,sharey=True)
    axes1 = axes1.ravel()
    axes = axes + list(axes1)

for counter,receptors in enumerate (data_cluster.index):
    alt_TLR = data_50_scaffolds.loc[receptors]
    r_pearson = alt_TLR.corr(WT_ref)
    
    axes[counter].scatter(WT_ref[0:50],alt_TLR[0:50],s=120,edgecolors='k',c=Color_length,marker='o')
    axes[counter].scatter(WT_ref[50:100],alt_TLR[50:100],s=120,edgecolors='k',c=Color_length,marker='s')
    axes[counter].scatter(WT_ref[100:150],alt_TLR[100:150],s=120,edgecolors='k',c=Color_length,marker='^')
    axes[counter].scatter(WT_ref[150:200],alt_TLR[150:200],s=120,edgecolors='k',c=Color_length,marker='*')
    axes[counter].plot(x,x,':k')
    axes[counter].plot(x,y_thres,':k',linewidth = 0.5)
    axes[counter].plot(y_thres,x,':k',linewidth = 0.5)
#    axes[counter].set_aspect('equal')
    axes[counter].set_xlim([low_lim,high_lim])
    axes[counter].set_ylim([low_lim,high_lim])
#    flat_axes[counter].xticks(list(range(-14,-4,2)))
#    flat_axes[counter].yticks(list(range(-14,-4,2)))
#    flat_axes[counter].tick_params(axis='both', which='major', labelsize=24)
    
#    axes[counter].set_xlabel('$\Delta$G$^{11ntR}_{bind}$ (kcal/mol)')
#    axes[counter].set_ylabel('$\Delta$G$^{mut}_{bind}$ (kcal/mol)')
    axes[counter].set_title(alt_TLR.name)
    r_str = 'r = ' + '{0:.2f}'.format(r_pearson)
    axes[counter].text(-10,-14,r_str)
plt.tight_layout()
#fig1.savefig('figure1.pdf')
#fig2.savefig('figure2.pdf')
#%%
# for each of the cluster, calculate the potassium effect 
next_cluster = 'cluster_1'
data_cluster = all_clusters_nan[next_cluster].copy()
data_columns = list(data_cluster.columns)
for receptors in data_cluster.index:
    data_receptor = data_cluster.loc[receptors]
    data_5Mg = data_receptor[100:150]
    data_5Mg[data_5Mg == -7.1] = np.nan
    data_5Mg150K = data_receptor[150:200]
    data_5Mg150K[data_5Mg150K == -7.1] = np.nan
    ddG_potassium = data_5Mg150K - data_5Mg.values
    plt.figure()
    plt.scatter(range(50),ddG_potassium)
#%%
# for each of the cluster, calculate magnesium decresase effect 
next_cluster = 'cluster_3'
data_cluster = all_clusters_nan[next_cluster].copy()
data_columns = list(data_cluster.columns)
for receptors in data_cluster.index:
    data_receptor = data_cluster.loc[receptors]
    data_5Mg = data_receptor[100:150]
    data_5Mg[data_5Mg == -7.1] = np.nan
    data_30Mg = data_receptor[0:50]
    data_30Mg[data_30Mg == -7.1] = np.nan
    ddG_Mg = data_30Mg - data_5Mg.values
    plt.figure()
    plt.scatter(range(50),ddG_Mg)    
#%%    
# for each of the cluster, calculate GAAA vs GUAA decresase effect 
next_cluster = 'cluster_1'
data_cluster = all_clusters_nan[next_cluster].copy()
data_columns = list(data_cluster.columns)
for receptors in data_cluster.index:
    data_receptor = data_cluster.loc[receptors]
    data_GUAA = data_receptor[50:100]
    data_GUAA[data_GUAA == -7.1] = np.nan
    data_30Mg = data_receptor[0:50]
    data_30Mg[data_30Mg == -7.1] = np.nan
    ddG_GUAA = data_30Mg - data_GUAA.values
    plt.figure()
    plt.scatter(range(50),ddG_GUAA)      
#%%
#create labels ordered as clustergram
TLR_labels = pd.DataFrame()
TLR_labels['names'] = clustered_data.index
TLR_labels['cluster'] = list(clustered_data.cluster)
TLR_labels['type'] = list(clustered_data.receptor_type)
TLR_labels = TLR_labels.reindex(cg_pca_col.dendrogram_row.reordered_ind)
