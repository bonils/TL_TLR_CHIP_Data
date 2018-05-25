#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 13:07:41 2018

@author: Steve
"""

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
num_neighbors = 20
#%%
'''---------------Import data ------------------'''
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
tecto_data = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
sublib0 = tecto_data[tecto_data['sublibrary'] == 'tertcontacts_0']
print('original size of sublibrary 0: ',sublib0.shape)
sublib0 = sublib0.drop_duplicates(subset='seq', keep="last")
print('after deleting duplicates: ',sublib0.shape)
#%%
receptors_sublib0 = sorted(list(set(sublib0['r_name'])))
scaffolds_length = pd.concat([sublib0['old_idx'],sublib0['length']],axis=1)
scaffolds_length = scaffolds_length.drop_duplicates()
scaffolds_length = scaffolds_length.sort_values(by=['length'])
scaffolds_sublib0 = list(scaffolds_length['old_idx'])
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
    

conditions = ['dG_Mut2_GAAA_5mM_150mMK_1']
column_labels = ['dG_5Mg150K_GUAA']
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
                               data_50_scaffolds_5Mg150K],axis = 1)
#%%
prep_data,original_nan = prep_data_for_clustering_ver2(data_50_scaffolds,
                                                       dG_threshold,dG_replace,nan_threshold,
                                                       num_neighbors=10)
prep_data_with_nan = prep_data.copy()
prep_data_with_nan[original_nan] = np.nan
#%%
'''------------PCA Analysis of raw data-----------------'''
pca,transformed,loadings = doPCA(prep_data)
#plot explained variance by PC
pd.Series(pca.explained_variance_ratio_).plot(kind='bar')
plt.ylabel('fraction of variance \nexplained by each PC', fontsize=14)
plt.tight_layout()
num_PCA = 4
print('Fraction explained by the first ',str(num_PCA), 'PCAs :',sum(pca.explained_variance_ratio_[:num_PCA]))


list_PCAs = list(transformed.columns[:num_PCA])
z_pca = sch.linkage(transformed.loc[:,list_PCAs],method='ward') 
cg_pca = sns.clustermap(prep_data_with_nan,row_linkage=z_pca, col_cluster=False
                        , vmin=-12,vmax=-7.1)
X = transformed.loc[:,list_PCAs]
c, coph_dists = cophenet(z_pca, pdist(X))
print('cophenetic distance: ',c)
plt.show()

sch.dendrogram(z_pca,color_threshold=15)
max_d = 15
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

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)    
cg_pca_col = sns.clustermap(prep_data_with_nan,row_linkage=z_pca,
                            col_cluster=False, vmin=-15,vmax=-7.1,row_colors=row_color)
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

#cg_pca_col.savefig('clustermap.svg')  
#%%
'''------------PCA Analysis of normalized data-----------------'''
#(3) Subtract the mean per row
norm_data = prep_data.sub(prep_data.mean(axis = 1),axis = 0)
norm_data_nan = norm_data.copy()
norm_data_nan[original_nan] = np.NAN
#%%
pca,transformed,loadings = doPCA(norm_data)
#plot explained variance by PC
pd.Series(pca.explained_variance_ratio_).plot(kind='bar')
plt.ylabel('fraction of variance \nexplained by each PC', fontsize=14)
plt.tight_layout()
num_PCA = 8
print('Fraction explained by the first ',str(num_PCA), 'PCAs :',sum(pca.explained_variance_ratio_[:num_PCA]))
#%%
list_PCAs = list(transformed.columns[:num_PCA])
z_pca = sch.linkage(transformed.loc[:,list_PCAs],method='ward') 
cg_pca = sns.clustermap(norm_data_nan,row_linkage=z_pca, col_cluster=False
                        , vmin=-3,vmax=3,cmap='coolwarm')
X = transformed.loc[:,list_PCAs]
c, coph_dists = cophenet(z_pca, pdist(X))
print('cophenetic distance: ',c)
plt.show()
#%%
sch.dendrogram(z_pca,color_threshold=15)
max_d = 15
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
cg_pca_col = sns.clustermap(norm_data_nan,row_linkage=z_pca,
                            col_cluster=False, vmin=-3,vmax=3,cmap='coolwarm',row_colors=row_color)
#%%
sublib0.to_csv('sublib0.csv')





