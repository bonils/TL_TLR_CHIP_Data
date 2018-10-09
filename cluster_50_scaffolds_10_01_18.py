#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 13:27:12 2018

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
from sklearn.metrics.pairwise import euclidean_distances
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
high_lim = -6

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

AC_data = grouped_lib.get_group('UAUGG_CCUACG')
AC_data = AC_data.set_index('old_idx')
AC_data = AC_data.reindex(WT_data.index)
#%%
A = pd.concat([WT_data['dG_Mut2_GAAA'],AC_data['dG_Mut2_GAAA']],axis=1)
A_avg = A.mean(axis=1)
A_sort = A_avg.sort_values()
#%%
data_5_scaff = pd.read_pickle("./prep_data_with_nan_09_30_18.pkl")
receptor_info = pd.read_pickle('./receptor_info.pkl')
receptors_types_matrix = pd.read_pickle('./receptors_types_matrix.pkl')
#%%
selected_receptors = list(data_5_scaff.index)
#%% get only selected receptors from main data
selected_data = entire_lib_selected[entire_lib_selected['r_seq'].isin(selected_receptors)]
#%%
sublib0 = selected_data[selected_data['sublibrary'] == 'tertcontacts_0']
sublib1 = selected_data[selected_data['sublibrary'] == 'tertcontacts_1']
sublib2 = selected_data[selected_data['sublibrary'] == 'tertcontacts_2']
sublib4 = selected_data[selected_data['sublibrary'] == 'tertcontacts_4']
sublib5 = selected_data[selected_data['sublibrary'] == 'tertcontacts_5']
#%%
# get data for which there are 50 scaffolds 
data_to_cluster = pd.concat([sublib0,sublib2])
#%%
fifty_scaffolds = list(set(data_to_cluster['old_idx']))
fifty_scaffolds = list(A_sort.index)

#%% Start analyzing only five scaffolds, create dataframe with all data (5scaffolds * conditions)
# for eact TLR ina a single row.
scaffolds_five = fifty_scaffolds
unique_receptors = list(set(data_to_cluster['r_seq']))
data = data_to_cluster
conditions = ['dG_Mut2_GAAA']#, 'dG_Mut2_GUAA_1']
column_labels = ['dG_30mM_Mg_GAAA']#,'dG_30mM_Mg_GUAA']
row_index= ('r_seq')
flanking = 'normal'

data_50_scaffolds_GAAA = pd.DataFrame(index = unique_receptors)

for scaffolds in scaffolds_five:
    
    next_scaffold = str(scaffolds)
    next_column_labels = [name + '_' +  next_scaffold for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_GAAA = pd.concat([data_50_scaffolds_GAAA,next_df],axis = 1)

conditions = ['dG_Mut2_GUAA_1']
column_labels = ['dG_30mM_Mg_GUAA']
row_index= ('r_seq')
flanking = 'normal'

data_50_scaffolds_GUAA = pd.DataFrame(index = unique_receptors)

for scaffolds in scaffolds_five:
    
    next_scaffold = str(scaffolds)
    next_column_labels = [name + '_' +  next_scaffold for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_GUAA = pd.concat([data_50_scaffolds_GUAA,next_df],axis = 1)


conditions = ['dG_Mut2_GAAA_5mM_2']
column_labels = ['dG_5mM_Mg_GAAA']
row_index= ('r_seq')
flanking = 'normal'

data_50_scaffolds_5Mg = pd.DataFrame(index = unique_receptors)

for scaffolds in scaffolds_five:
    
    next_scaffold = str(scaffolds)
    next_column_labels = [name + '_' +  next_scaffold for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_5Mg = pd.concat([data_50_scaffolds_5Mg,next_df],axis = 1)


conditions = ['dG_Mut2_GAAA_5mM_150mMK_1']
column_labels = ['dG_5Mg150K_GAAA']
row_index= ('r_seq')
flanking = 'normal'

data_50_scaffolds_5Mg150K = pd.DataFrame(index = unique_receptors)

for scaffolds in scaffolds_five:
    
    next_scaffold = str(scaffolds)
    next_column_labels = [name + '_' +  next_scaffold for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(unique_receptors)
    data_50_scaffolds_5Mg150K = pd.concat([data_50_scaffolds_5Mg150K,next_df],axis = 1)
    
data_50_scaffolds = pd.concat([data_50_scaffolds_GAAA,
                               data_50_scaffolds_5Mg,
                               data_50_scaffolds_5Mg150K,
                               data_50_scaffolds_GUAA],axis = 1)
#%%  
data_30mMMg = data_50_scaffolds[list(data_50_scaffolds.columns[0:50])+list(data_50_scaffolds.columns[150:200])]
conditions = (data_30mMMg.isna()) | (data_30mMMg > -7.1)
#how many of the columns are below threshold or nan
number_condition = conditions.sum(axis=1)
limit = number_condition > 40
data_50_scaffolds_filtered = data_50_scaffolds[~limit]   
#add Vc2: this is an important motif I want to keep here

#%%
###########################################################################################
#%  THIS IS FOR CLUSTERING GAAA and GUAA as different TL/TLRs
B = WT_data.copy()
C = B.groupby('length')
s = []
for length in C.groups:
    next_group = C.get_group(length)
    next_group = next_group.sort_values('dG_Mut2_GAAA')
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
data_50_scaffolds_GAAA_2 = data_50_scaffolds_GAAA.copy()   
data_50_scaffolds_GUAA_2 = data_50_scaffolds_GUAA.copy() 
data_50_scaffolds_GAAA_2.columns = fifty_scaffolds
data_50_scaffolds_GUAA_2.columns = fifty_scaffolds

data_50_scaffolds_GAAA_2 = data_50_scaffolds_GAAA_2[fifty_scaffolds_bylength]
data_50_scaffolds_GUAA_2 = data_50_scaffolds_GUAA_2[fifty_scaffolds_bylength]

data_50_scaffolds_GAAA_2 = data_50_scaffolds_GAAA_2.reindex(data_50_scaffolds_filtered.index)
data_50_scaffolds_GUAA_2 = data_50_scaffolds_GUAA_2.reindex(data_50_scaffolds_filtered.index)
index_GAAA = ['GAAA_' + sequence for sequence in data_50_scaffolds_GAAA_2.index]
index_GUAA = ['GUAA_' + sequence for sequence in data_50_scaffolds_GUAA_2.index]
data_50_scaffolds_GAAA_2.index = index_GAAA
data_50_scaffolds_GUAA_2.index = index_GUAA

data_TLTLR = pd.concat([data_50_scaffolds_GAAA_2,data_50_scaffolds_GUAA_2])
TL = pd.DataFrame(index = data_TLTLR.index)
TL['GAAA'] = [1] * len(data_50_scaffolds_GAAA_2) + [0]*len(data_50_scaffolds_GAAA_2)
TL['GUAA'] = [0] * len(data_50_scaffolds_GAAA_2) + [1]*len(data_50_scaffolds_GAAA_2)
prep_data_TLTLR,original_nan_TLTLR = prep_data_for_clustering_ver3(data_TLTLR,
                                                       dG_threshold,dG_replace,nan_threshold,
                                                       num_neighbors=10)

TL = TL.reindex(prep_data_TLTLR.index)
# reinsert nan values VERY IMPORTANT!!!!!
# Heatmaps and many of the other plots need not to included nan values
# nans were assigned values by interpolation only for clustering and other 
# statistical tools that do not support missing values.
prep_data_TLTLR_with_nan = prep_data_TLTLR.copy()
prep_data_TLTLR_with_nan[original_nan_TLTLR] = np.nan


prep8 = prep_data_TLTLR_with_nan[sc8].median(axis=1)
prep9 = prep_data_TLTLR_with_nan[sc9].median(axis=1)
prep10 = prep_data_TLTLR_with_nan[sc10].median(axis=1)
prep11 = prep_data_TLTLR_with_nan[sc11].median(axis=1)

prep_avg_length = pd.concat([prep8,prep9,prep10,prep11],axis=1)
prep_avg_length.columns = ['8','9','10','11']
prep_avg_length_norm = prep_avg_length.sub(prep_avg_length.mean(axis=1),axis=0)
#z_pca = sch.linkage(prep_avg_length_norm,method='ward',optimal_ordering=True) 
#cg_pca = sns.clustermap(prep_avg_length_norm,row_linkage=z_pca, col_cluster=False
#                        ,row_cluster=True, vmin=-1.5,vmax=1.5,cmap='coolwarm')
#%%
num_PCA = 2
pca,transformed,loadings = doPCA(prep_avg_length_norm)
#plot explained variance by PC
plt.figure()
pd.Series(pca.explained_variance_ratio_).plot(kind='bar')
plt.ylabel('fraction of variance \nexplained by each PC', fontsize=14)
plt.tight_layout()
print('Fraction explained by the first ',str(num_PCA), 'PCAs :',sum(pca.explained_variance_ratio_[:num_PCA]))
list_PCAs = list(transformed.columns[:num_PCA])
z_pca = sch.linkage(transformed.loc[:,list_PCAs],method='ward',optimal_ordering=True) 

#plot clustermap of medians by length
cg_pca = sns.clustermap(prep_avg_length_norm,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=-1.5,vmax=1.5,cmap='coolwarm')
X = transformed.loc[:,list_PCAs]
c, coph_dists = cophenet(z_pca, pdist(X))
print('cophenetic distance: ',c)

#plot clustermap of TL identities
cg_pca = sns.clustermap(TL,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=0,vmax=1,cmap='Greys')
#cg_pca.savefig('/Volumes/NO NAME/Clustermaps/fifty_scaff_TL_03_10_2018.svg')

avg_per_length = prep_avg_length.mean(axis=1)

cg_pca = sns.clustermap(avg_per_length,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=-12,vmax=-6,cmap='Blues_r')
#cg_pca.savefig('/Volumes/NO NAME/Clustermaps/fifty_scaff_avg_03_10_2018.svg')


distance_threshold = 2.5
plt.plot()
#sch.dendrogram(z_pca,color_threshold=distance_threshold)
max_d = distance_threshold
clusters = fcluster(z_pca, max_d, criterion='distance')
number_clusters = max(clusters)
print('number of clusters based on distance of ',str(max_d), ':', str(number_clusters))
# Append cluster number to dataframe with data
clustered_data = prep_avg_length_norm.copy()
cluster_series = pd.Series(clusters,index=clustered_data.index)
clustered_data['cluster'] = cluster_series
#Replot clustergram with rows colored as above
row_color = pd.Series(clusters,index=clustered_data.index)
cluster_colors = ['g','r','c','m','y','k','b','orange','g','r','c','m','y','k','b']
num_clusters = len(cluster_series.unique())
num_clusters
for i in range(num_clusters):
    row_color[row_color == (i+1)] = cluster_colors[i]
cg_pca = sns.clustermap(prep_avg_length_norm,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=-1.5,vmax=1.5,cmap='coolwarm',
                        row_colors = row_color)
#cg_pca.savefig('/Volumes/NO NAME/Clustermaps/fifty_scaff_avg_length_more_clusters_03_10_2018.svg')

receptors_with_TL = pd.DataFrame(index = prep_avg_length_norm.index)
types = []
for receptor in receptors_with_TL.index:
    next_receptor = receptor[5:]
    types.append(receptor_info.loc[next_receptor]['type'])
receptors_with_TL['type'] = types 
TLRs_matrix = pd.DataFrame(index = receptors_with_TL.index)
TLRs_matrix['11ntR'] = receptors_with_TL == '11ntR'
TLRs_matrix['IC3'] = receptors_with_TL == 'IC3'
TLRs_matrix['VC2'] = receptors_with_TL == 'VC2'
TLRs_matrix['in_vitro'] = receptors_with_TL == 'in_vitro'
TLRs_matrix['intermediates'] = receptors_with_TL == 'other'
cg_pca = sns.clustermap(TLRs_matrix,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=0,vmax=1,cmap='Greys')
#cg_pca.savefig('/Volumes/NO NAME/Clustermaps/fifty_scaff_avg_length_TLRS_03_10_2018.svg')
#%%
clustered_data['type'] = receptors_with_TL['type']


v1 = 'GUAGG_CCUAAAC'
v2 = 'UAUGG_CCUAAG'

variant1 = grouped_lib.get_group(v1)
variant1 = variant1.set_index('old_idx')
variant1 = variant1.reindex(fifty_scaffolds)
variant2 = grouped_lib.get_group(v2)
variant2 = variant2.set_index('old_idx')
variant2 = variant2.reindex(fifty_scaffolds)
plt.figure()
plt.scatter(variant2['dG_Mut2_GAAA'],variant1['dG_Mut2_GAAA'])
plt.xlim(-14,-6)
plt.ylim(-14,-6)

plt.figure()
plt.scatter(variant2['dG_Mut2_GAAA'],variant1['dG_Mut2_GUAA_1'])
plt.xlim(-14,-6)
plt.ylim(-14,-6)

plt.figure()
plt.scatter(variant1['dG_Mut2_GAAA'],variant1['dG_Mut2_GUAA_1'])
plt.xlim(-14,-6)
plt.ylim(-14,-6)

print(clustered_data.loc['GUAA_GUAGG_CCUAAAC']['cluster'])
print(clustered_data.loc['GAAA_GUAGG_CCUAAAC']['cluster'])

#%% CLUSTER ENTIRE SET OF dGs INSTEAD OF MEDIANS PER LENGTH BUT STILL PLOT MEDIANS

#Because we are more concerned with the behavior of the TLR as the conditions
#and the scaffolds are changed, we substract the average of the entire row.
#Normalized each tetraloop-receptor with respect to its mean accross columns
#mean_per_row = prep_data_TLTLR.mean(axis=1)
#prep_data_TLTLR_norm = prep_data_TLTLR.copy()
#prep_data_TLTLR_norm = prep_data_TLTLR_norm.sub(mean_per_row,axis=0)
#prep_data_TLTLR_norm_with_nan = prep_data_TLTLR_norm.copy()
#prep_data_TLTLR_norm_with_nan[original_nan_TLTLR] = np.nan

#
#How many PCs shall we use????
#num_PCA = 23
#pca,transformed,loadings = doPCA(prep_data_TLTLR_norm)
#
##plot explained variance by PC
#plt.figure()
#pd.Series(pca.explained_variance_ratio_).plot(kind='bar')
#plt.ylabel('fraction of variance \nexplained by each PC', fontsize=14)
#plt.tight_layout()
#print('Fraction explained by the first ',str(num_PCA), 'PCAs :',sum(pca.explained_variance_ratio_[:num_PCA]))
#list_PCAs = list(transformed.columns[:num_PCA])
#z_pca = sch.linkage(transformed.loc[:,list_PCAs],method='ward',optimal_ordering=True) 
#cg_pca = sns.clustermap(prep_data_TLTLR_norm_with_nan,row_linkage=z_pca, col_cluster=False
#                        ,row_cluster=True, vmin=-3.2,vmax=3.2,cmap='coolwarm')
#
#X = transformed.loc[:,list_PCAs]
#c, coph_dists = cophenet(z_pca, pdist(X))
#print('cophenetic distance: ',c)
#
#cg_pca = sns.clustermap(prep_avg_length_norm,row_linkage=z_pca, col_cluster=False
#                        ,row_cluster=True, vmin=-1.5,vmax=1.5,cmap='coolwarm')
#
#cg_pca = sns.clustermap(TL,row_linkage=z_pca, col_cluster=False
#                        ,row_cluster=True, vmin=0,vmax=1,cmap='Greys')
##
#22
#distance_threshold = 8
#sch.dendrogram(z_pca,color_threshold=distance_threshold)
#max_d = distance_threshold
#clusters = fcluster(z_pca, max_d, criterion='distance')
#number_clusters = max(clusters)
#print('number of clusters based on distance of ',str(max_d), ':', str(number_clusters))
## Append cluster number to dataframe with data
#clustered_data = prep_avg_length.copy()
#cluster_series = pd.Series(clusters,index=clustered_data.index)
#clustered_data['cluster'] = cluster_series
##Replot clustergram with rows colored as above
#row_color = pd.Series(clusters,index=clustered_data.index)
#cluster_colors = ['g','r','c','m','y','k','b','orange','g','r','c','m','y','k','b']
#num_clusters = len(cluster_series.unique())
#num_clusters
#for i in range(num_clusters):
#    row_color[row_color == (i+1)] = cluster_colors[i]
#cg_pca = sns.clustermap(prep_avg_length_norm,row_linkage=z_pca, col_cluster=False
#                        ,row_cluster=True, vmin=-1.5,vmax=1.5,cmap='coolwarm',
#                        row_colors = row_color)
#####################################################################################
#%% BACK TO ORGINAL ANALYSIS   
prep_data,original_nan = prep_data_for_clustering_ver3(data_50_scaffolds_filtered,
                                                       dG_threshold,dG_replace,nan_threshold,
                                                       num_neighbors=10)
# reinsert nan values VERY IMPORTANT!!!!!
# Heatmaps and many of the other plots need not to included nan values
# nans were assigned values by interpolation only for clustering and other 
# statistical tools that do not support missing values.
prep_data_with_nan = prep_data.copy()
prep_data_with_nan[original_nan] = np.nan
#Because we are more concerned with the behavior of the TLR as the conditions
#and the scaffolds are changed, we substract the average of the entire row.
#Normalized each tetraloop-receptor with respect to its mean accross columns
mean_per_row = prep_data.mean(axis=1)
prep_data_norm = prep_data.copy()
prep_data_norm = prep_data_norm.sub(mean_per_row,axis=0)
prep_data_norm_with_nan = prep_data_norm.copy()
prep_data_norm_with_nan[original_nan] = np.nan
#%%
#CLUSTER THE DATA AGAIN BUT NOW WITH PCA ANALYSIS PRECEEDING THE CLUSTERING;
#FOR THE PCA, I AM USING THE FUNCTION THAT SARAH GAVE ME
'''------------PCA Analysis of raw data-----------------'''

#How many PCs shall we use????
num_PCA = 35
pca,transformed,loadings = doPCA(prep_data_norm)
#plot explained variance by PC
plt.figure()
pd.Series(pca.explained_variance_ratio_).plot(kind='bar')
plt.ylabel('fraction of variance \nexplained by each PC', fontsize=14)
plt.tight_layout()
print('Fraction explained by the first ',str(num_PCA), 'PCAs :',sum(pca.explained_variance_ratio_[:num_PCA]))
#%%
plt.figure()
cumulative = 1 - np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(len(cumulative)),cumulative)

list_PCAs = list(transformed.columns[:num_PCA])
#%%
z_pca = sch.linkage(transformed.loc[:,list_PCAs],method='ward',optimal_ordering=True) 
cg_pca = sns.clustermap(prep_data_norm_with_nan,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=-3.2,vmax=3.2,cmap='coolwarm')
#cg_pca.savefig('/Volumes/NO NAME/Clustermaps/fifty_scaff_02_10_2018.svg')
#%%
z_pca = sch.linkage(transformed.loc[:,list_PCAs],method='ward')#,optimal_ordering=True) 
cg_pca = sns.clustermap(prep_data_with_nan,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True)#, vmin=-3.2,vmax=3.2,cmap='coolwarm')
#%%
distance_threshold = 22
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
#
#Replot clustergram with rows colored as above
row_color = pd.Series(clusters,index=clustered_data.index)
cluster_colors = ['g','r','c','m','y','k','b','orange','g','r','c','m','y','k','b']
num_clusters = len(cluster_series.unique())
num_clusters
for i in range(num_clusters):
    row_color[row_color == (i+1)] = cluster_colors[i]
 
cg_pca_col = sns.clustermap(prep_data_with_nan,row_linkage=z_pca,
                            col_cluster=False,row_cluster=True,row_colors=row_color,)

cg_pca = sns.clustermap(prep_data_norm_with_nan,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=-3.2,vmax=3.2,cmap='coolwarm',
                        row_colors = row_color)
cg_pca.savefig('/Volumes/NO NAME/Clustermaps/fifty_scaff_06_10_2018.svg')
#
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
cluster_to_plot = 'cluster_1'
#WT_ref = data_50_scaffolds.loc['11ntR'] 
WT_ref = prep_data_with_nan.loc['UAUGG_CCUAAG']
next_cluster = all_clusters_nan[cluster_to_plot]
S1,S2 = next_cluster.shape
rand_index = np.random.randint(1,S1)
alternative_TLR = next_cluster.index[rand_index]
alt_TLR = prep_data_with_nan.loc[alternative_TLR]
r_pearson = alt_TLR.corr(WT_ref)

#%Colors to used for scatter plotting
receptors_sublib0 = sorted(list(set(sublib0['r_name'])))
scaffolds_length = pd.concat([sublib0['old_idx'],sublib0['length']],axis=1)
scaffolds_length = scaffolds_length.drop_duplicates()
scaffolds_length = scaffolds_length.sort_values(by=['length'])
scaffolds_sublib0 = list(scaffolds_length['old_idx'])
scaffolds_length = scaffolds_length.set_index('old_idx')
Color_length =scaffolds_length['length'].copy()
Color_length[Color_length == 8] = 'magenta'
Color_length[Color_length == 9] = 'cyan'
Color_length[Color_length == 10] = 'brown'
Color_length[Color_length == 11] = 'black'


x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
#x_ddG = [ddG_average + x[0],ddG_average + x[1]]
#plt.plot(x,x_ddG,'--r',linewidth = 3)
plt.scatter(WT_ref[0:50],alt_TLR[0:50],s=120,edgecolors='k',c=Color_length,marker='o')
plt.scatter(WT_ref[50:100],alt_TLR[50:100],s=120,edgecolors='k',c=Color_length,marker='s')
plt.scatter(WT_ref[100:150],alt_TLR[100:150],s=120,edgecolors='k',c=Color_length,marker='^')
plt.scatter(WT_ref[150:200],alt_TLR[150:200],s=180,edgecolors='k',c=Color_length,marker='*')
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


cluster1 = clustered_data[clustered_data['cluster'] == 2].index
cluster1_data = prep_data_norm.loc[cluster1]
C = cluster1_data.transpose()
BB = euclidean_distances(C)
print(BB.mean())
#%%
receptor = 'UAUGG_CCUAAG'
receptor_data = grouped_lib.get_group(receptor)
by_length = receptor_data.groupby('length')
#%% get medians by lenght for GAAA and GUAA
GAAA_by_length = pd.DataFrame(0,index = prep_data.index, columns = [8.0,9.0,10.0,11.0])
for receptor in prep_data.index:
    receptor_data = grouped_lib.get_group(receptor)
    by_length = receptor_data.groupby('length')
    GAAA_by_length.loc[receptor] = by_length['dG_Mut2_GAAA'].median()


    
GUAA_by_length = pd.DataFrame(0,index = prep_data.index, columns = [8.0,9.0,10.0,11.0])
for receptor in prep_data.index:
    receptor_data = grouped_lib.get_group(receptor)
    by_length = receptor_data.groupby('length')
    GUAA_by_length.loc[receptor] = by_length['dG_Mut2_GUAA_1'].median() 
    
    
median_GAAA = GAAA_by_length.median(axis =1)
GAAA_by_length_norm = GAAA_by_length.sub(median_GAAA,axis = 0)    
    
median_GUAA = GUAA_by_length.median(axis =1)
GUAA_by_length_norm = GUAA_by_length.sub(median_GUAA,axis = 0) 

#%%
cg_length= sns.clustermap(GUAA_by_length,row_linkage=z_pca, col_cluster=False,
                        row_colors = row_color)#, cmap = 'coolwarm',vmin = -1.5,vmax = 1.5)

#%%
cg_length= sns.clustermap(GUAA_by_length_norm,row_linkage=z_pca, col_cluster=False,
                        row_colors = row_color, cmap = 'coolwarm',vmin = -1.5,vmax = 1.5)
#cg_length.savefig('/Volumes/NO NAME/Clustermaps/GUAA_length_06_10_2018.svg')

cg_length= sns.clustermap(GAAA_by_length_norm,row_linkage=z_pca, col_cluster=False,
                        row_colors = row_color, cmap = 'coolwarm',vmin = -1.5,vmax = 1.5)
cg_length.savefig('/Volumes/NO NAME/Clustermaps/GAAA_length_06_10_2018.svg')
#%% Plot recepto types based on clustering above
receptors_types = receptors_types_matrix.copy()
receptors_types = receptors_types.reindex(prep_data_norm_with_nan.index)
cg_receptors= sns.clustermap(receptors_types,row_linkage=z_pca, col_cluster=False,
                        vmin=0,vmax=1,cmap='Greys',row_colors = row_color)
cg_receptors.savefig('/Volumes/NO NAME/Clustermaps/receptors_06_10_2018.svg')
#%% Plot specificty based on clustering above
#calculate average TL specificity
GAAA = prep_data_with_nan[prep_data_with_nan.columns[0:50]].copy()
GUAA = prep_data_with_nan[prep_data_with_nan.columns[150:200]].copy()
GAAA.columns = range(50)
GUAA.columns = range(50)
GAAA_limits = GAAA == -7.1
GUAA_limits = GUAA == -7.1
bad_comparison = GUAA_limits & GAAA_limits
specificity = GAAA.subtract(GUAA)
specificity[bad_comparison] = np.nan
mean_specificity = specificity.mean(axis=1)
cg_pca = sns.clustermap(mean_specificity,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=-3.5,vmax=1,cmap='coolwarm',
                        row_colors = row_color)
cg_pca.savefig('/Volumes/NO NAME/Clustermaps/specificity_02_10_2018.svg')
#%%
#calculate spearman R relate to WT 
WT_GAAA = GAAA.loc['UAUGG_CCUAAG']
variant = GAAA.loc['UAUGG_CCUACG']

GAAA_for_corr = GAAA.copy()
GAAA_for_corr[GAAA_limits] = np.nan
nan_counts = GAAA_for_corr.isna().sum(axis=1)
r_values = []
for receptors in GAAA_for_corr.index:
    r_values.append(WT_GAAA.corr(GAAA_for_corr.loc[receptors]))
    
r_df = pd.DataFrame(r_values,index=GAAA_for_corr.index)
r_sq = r_df **2


#r_sq[nan_counts > 25] = np.nan

cg_pca = sns.clustermap(r_sq,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=0,vmax=1,cmap='Blues',
                        row_colors = row_color)

n_values = 50 - nan_counts 
cg_pca = sns.clustermap(n_values,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=10,vmax=50,cmap='Blues',
                        row_colors = row_color)


plt.figure()
plt.hist(r_sq.dropna().values)
plt.figure()
plt.hist(n_values.dropna().values)
#%%
#control
variant = GAAA_for_corr.loc['UAGGG_CCUAAG']
WT = GAAA_for_corr.loc['UAUGG_CCUAAG']
control = pd.concat([variant,WT],axis=1)
control.columns = ['variant','WT']
#control = control[control['variant']>-9]

control = control.dropna()

indices = list(control.index)
n = 5
random.shuffle(indices)
idx = indices[0:n]
x = control.loc[idx]
print(x)
r = x['variant'].corr(x['WT'])
print(r)
#%%
WT_GAAA = GAAA_for_corr.loc['UAUGG_CCUAAG']
ddG = variant.subtract(WT_GAAA)
ddG_avg = ddG.median()
ddG_model = WT_GAAA + ddG_avg


rmse = []
for receptor in GAAA_for_corr.index:
    variant = GAAA_for_corr.loc[receptor]
    ddG = variant.subtract(WT_GAAA)
    ddG_avg = ddG.median()
    ddG_model = WT_GAAA + ddG_avg
    model = pd.concat([variant,ddG_model],axis=1)
    model.columns = ['data','model']
    model = model.dropna()
    rms = sqrt(mean_squared_error(model['data'], model['model']))
    rmse.append(rms)
rmse_df = pd.DataFrame(rmse,index = GAAA_for_corr.index)    
cg_pca = sns.clustermap(rmse_df,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True,
                        row_colors = row_color,
                        vmin = 0, vmax = 1,cmap = 'Blues_r') 

#cg_pca.savefig('/Volumes/NO NAME/Clustermaps/fifty_scaff_rmse_02_10_2018.svg')   
#%%



#%%
model = pd.concat([variant,ddG_model],axis=1)
model.columns = ['data','model']
model = model.dropna()
rms = sqrt(mean_squared_error(model['data'], model['model']))
print(rms)
#print(ddG_avg)
#%%
#calculate potssium effect
Mg5 = prep_data_with_nan[prep_data_with_nan.columns[50:100]].copy()
Mg5K150 = prep_data_with_nan[prep_data_with_nan.columns[100:150]].copy()
Mg5.columns = range(50)
Mg5K150.columns = range(50)


Mg5_limits = Mg5 == -7.1
Mg5K150_limits = Mg5K150 == -7.1
bad_comparison = Mg5_limits & Mg5K150_limits


potassium_diff = Mg5K150.subtract(Mg5)

potassium_diff[bad_comparison] = np.nan
nan_counts = potassium_diff.isna().sum(axis=1)

potassium_mean = potassium_diff.mean(axis=1)
potassium_mean[nan_counts >= 45] = np.nan

#at least 3 measurements

cg_pca = sns.clustermap(potassium_mean,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=-1.5,vmax=1.5,cmap='coolwarm',
                        row_colors = row_color)
cg_pca.savefig('/Volumes/NO NAME/Clustermaps/potassium_06_10_2018.svg')
#%%

stabilizing_potassium = potassium_mean < -0.2
stabilizing_potassium[nan_counts >= 45] = np.nan
cg_pca = sns.clustermap(stabilizing_potassium,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=0,vmax=1,cmap='Greys',
                        row_colors = row_color)

#
select_receptor_info = receptor_info.reindex(potassium_mean.index)
mutations_df = pd.DataFrame(index = select_receptor_info.index)
resi1 = []
resi2 = []
resi3 = []
resi4 = []
resi5 = []
resi6 = []
resi7 = []
resi8 = []
resi9 = []
resi10 = []
resi11 = []


for receptor in mutations_df.index:
    if len(receptor) == 12:
        if receptor[0] == 'U':
            resi1.append(0)
        else:
            resi1.append(1)
    
        if receptor[1] == 'A':
            resi2.append(0)
        else:
            resi2.append(1)            
    
        if receptor[2] == 'U':
            resi3.append(0)
        else:
            resi3.append(1)
    
        if receptor[3] == 'G':
            resi4.append(0)
        else:
            resi4.append(1)
    
        if receptor[4] == 'G':
            resi5.append(0)
        else:
            resi5.append(1)
    
        if receptor[6] == 'C':
            resi6.append(0)
        else:
            resi6.append(1)
    
        if receptor[7] == 'C':
            resi7.append(0)
        else:
            resi7.append(1)
    
        if receptor[8] == 'U':
            resi8.append(0)
        else:
            resi8.append(1)
    
        if receptor[9] == 'A':
            resi9.append(0)
        else:
            resi9.append(1)
    
        if receptor[10] == 'A':
            resi10.append(0)
        else:
            resi10.append(1)
    
        if receptor[11] == 'G':
            resi11.append(0)
        else:
            resi11.append(1)
    else:
        resi1.append(np.nan)
        resi2.append(np.nan)
        resi3.append(np.nan)
        resi4.append(np.nan)
        resi5.append(np.nan)
        resi6.append(np.nan)
        resi7.append(np.nan)
        resi8.append(np.nan)
        resi9.append(np.nan)
        resi10.append(np.nan)
        resi11.append(np.nan)  

#mutations_df['type'] = select_receptor_info['type']
mutations_df['mut_res1'] = resi1
mutations_df['mut_res2'] = resi2
mutations_df['mut_res3'] = resi3
mutations_df['mut_res4'] = resi4
mutations_df['mut_res5'] = resi5
mutations_df['mut_res6'] = resi6
mutations_df['mut_res7'] = resi7
mutations_df['mut_res8'] = resi8
mutations_df['mut_res9'] = resi9
mutations_df['mut_res10'] = resi10
mutations_df['mut_res11'] = resi11  

mutations_df2 = mutations_df.copy()
mutations_df2[mutations_df == 1] = 0
mutations_df2[mutations_df == 0] = 1

platform = pd.concat([mutations_df['mut_res9'],mutations_df['mut_res10']],axis=1)
wobble= pd.concat([mutations_df['mut_res11'],mutations_df['mut_res1']],axis=1)
core = platform = pd.concat([mutations_df['mut_res7'],mutations_df['mut_res8'],mutations_df['mut_res2'],mutations_df['mut_res4']],axis=1)
WC = pd.concat([mutations_df['mut_res6'],mutations_df['mut_res5']],axis=1)
bulge = mutations_df['mut_res3']
#%%
no_residues = [len(receptor) - 1 for receptor in mutations_df.index]
no_residues_df = pd.DataFrame(no_residues,index = mutations_df.index)
cg_pca = sns.clustermap(no_residues_df == 11,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=0,vmax=1,cmap='Greys_r',
                        row_colors = row_color)  
#%%
platform_nan = platform.copy()
platform_nan[platform == 1] = 0
platform_nan[platform.isna()] = 0.3
platform[nan_counts >= 45 ] = np.nan
platform[platform.isna()] = 0.8

#
cg_pca = sns.clustermap(platform,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=0,vmax=1,cmap='Greys_r',
                        row_colors = row_color)   
#%%
cg_pca = sns.clustermap(platform_nan,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=0,vmax=1,cmap='Greys',
                        row_colors = row_color)
#%%
mutations_df3 = mutations_df.copy()
mutations_df3[mutations_df.isna()] = 0.8
cg_pca = sns.clustermap(mutations_df3,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=0,vmax=1,cmap='Greys_r',
                        row_colors = row_color)      
#%%
length_8 = []
length_9 = []
length_10 = []
length_11 = []


for receptor in prep_data.index:
    receptor_data = grouped_lib.get_group(receptor)
    receptor_data = receptor_data.groupby('length')
    A = receptor_data['dG_Mut2_GAAA'].agg('median')
    length_8.append(A[8])
    length_9.append(A[9])
    length_10.append(A[10])
    length_11.append(A[11])
    
l_dict = {'8':length_8,'9':length_9,'10':length_10,'11':length_11}      
dG_length = pd.DataFrame(l_dict,index = prep_data.index)
dG_length_norm = dG_length.sub(dG_length['8'],axis=0)
cg_pca = sns.clustermap(dG_length_norm,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True,cmap ='coolwarm', 
                        row_colors = row_color) 



#%%
'''----------------Plot a random member of cluster n-------------------'''    
cluster_to_plot = 6
WT_ref = data_50_scaffolds.loc['UAUGG_CCUAAG'] 
next_cluster = clustered_data[clustered_data.cluster == cluster_to_plot]
S1,S2 = next_cluster.shape
rand_index = np.random.randint(1,S1)
alternative_TLR = next_cluster.index[rand_index]
alt_TLR = data_50_scaffolds.loc[alternative_TLR]
r_pearson = alt_TLR.corr(WT_ref)

Color_length = 'red'

x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
#x_ddG = [ddG_average + x[0],ddG_average + x[1]]
#plt.plot(x,x_ddG,'--r',linewidth = 3)
plt.scatter(WT_ref[0:50],alt_TLR[0:50],s=120,edgecolors='k',c=Color_length,marker='o')
plt.scatter(WT_ref[50:100],alt_TLR[50:100],s=120,edgecolors='k',c=Color_length,marker='s')
plt.scatter(WT_ref[100:150],alt_TLR[100:150],s=120,edgecolors='k',c=Color_length,marker='^')
plt.scatter(WT_ref[150:200],alt_TLR[150:200],s=180,edgecolors='k',c=Color_length,marker='*')
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
#plt.savefig(cluster_to_plot + '.svg')

#%%
#Plot GAAA vs GUAA for 11ntR-wt 
WTdata = grouped_lib.get_group('UAUGG_CCUAAG')
WTdata = WTdata.set_index('old_idx')
WT_GAAA = WTdata['dG_Mut2_GAAA']
WT_GUAA = WTdata['dG_Mut2_GUAA_1']
ddG_GUAA_wt = WT_GUAA.subtract(WT_GAAA)
plt.bar([0,1],[0,ddG_GUAA_wt.median()],yerr = [0,ddG_GUAA_wt.std()])
plt.ylim(-1,5)
#%%
plt.plot()

colors2 = Color_length.reindex(WT_GAAA.index)
plt.scatter(WT_GAAA,WT_GUAA,s=120,edgecolors='k',c=colors2,marker='o')
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

#%%
8bp = Color_length[Color_length == 'magenta']




