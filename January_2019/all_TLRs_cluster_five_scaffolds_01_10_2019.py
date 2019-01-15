#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:53:30 2018

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
#%
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
#%
#import data from csv files 
#Data has been separated into 11ntR, IC3, and in vitro
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
entire_lib = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
sublib5 = entire_lib[entire_lib['sublibrary'] == 'tertcontacts_5'].copy()
list_intermediates = set(sublib5.r_name)
print(len(list_intermediates))

duplicated_seq = entire_lib.duplicated(subset = 'seq') 
duplicated_data = entire_lib[duplicated_seq].copy()
duplicated_data['new_name'] = duplicated_data['r_name'] + '_' + duplicated_data['r_seq'] + '_' + duplicated_data['sublibrary']
unique_duplicated = set(list(duplicated_data['new_name']))
print(len(unique_duplicated))

# find which recpetors are duplicating with Vc2 to 11ntR intermediates
all_receptor_seq = set(entire_lib['r_seq'])
Vc2_inter = []
for i in list_intermediates:
    if ('11nt' in i) and ('Vc2' in i):
        Vc2_inter.append(i)
        
sublib5 = sublib5.set_index('r_name') 
 
Vc2_inter_data = sublib5.loc[Vc2_inter]
Vc2_inter_seq = set(Vc2_inter_data.r_seq)

dupl_Vc2_rec = []
for Vc2_rec in Vc2_inter_seq:
    for receptor in all_receptor_seq:
        if Vc2_rec == receptor:
            dupl_Vc2_rec.append(receptor)
            break
#%Drop repeats, just because they are there; is the same data
entire_lib = entire_lib.drop_duplicates(subset ='seq',keep = 'last')
#%Consider the ones only with normal closing base pair 
mask = entire_lib.b_name == 'normal' 
entire_lib_normal_bp = entire_lib[mask]
#%Exclude sublibrary 5 which has the mutation intermediates and cannot be easily classified (or maybe they can be classified as others)
mask = (entire_lib_normal_bp.sublibrary == 'tertcontacts_0') | (entire_lib_normal_bp.sublibrary == 'tertcontacts_1')|\
       (entire_lib_normal_bp.sublibrary == 'tertcontacts_2') | (entire_lib_normal_bp.sublibrary == 'tertcontacts_3')|\
       (entire_lib_normal_bp.sublibrary == 'tertcontacts_4') | (entire_lib_normal_bp.sublibrary == 'tertcontacts_5')
entire_lib_selected = entire_lib_normal_bp[mask].copy()
#%Create new name identifier because there are some that were given the same name
#but they are different receptors
#also attacht the sublibrary they came from for later reference
entire_lib_selected['new_name'] = entire_lib_selected.r_name + '_' + entire_lib_selected.r_seq #+ '_' + entire_lib_selected.sublibrary
#entire_lib_selected = entire_lib_selected.set_index('new_name')
#%
mask = entire_lib_selected['r_seq'] == 'UAUGG_CCUAAG'
canonical = entire_lib_selected[mask]
#%%
#------------------------------------------
apply_error_threshold = False
error_threshold = 1.5
#------------------------------------------

if apply_error_threshold:
    entire_lib_selected = entire_lib_selected.copy()
    mask = entire_lib_selected['dGerr_Mut2_GAAA'] > error_threshold
    entire_lib_selected['dG_Mut2_GAAA'][mask] = np.nan
    entire_lib_selected['dGerr_Mut2_GAAA'][mask] = np.nan
    print(sum(mask))
    
    mask = entire_lib_selected['dGerr_Mut2_GAAA_5mM_2'] > error_threshold
    entire_lib_selected['dG_Mut2_GAAA_5mM_2'][mask] = np.nan
    entire_lib_selected['dGerr_Mut2_GAAA_5mM_2'][mask] = np.nan
    print(sum(mask))
    
    mask = entire_lib_selected['dGerr_Mut2_GAAA_5mM_150mMK_1'] > error_threshold
    entire_lib_selected['dG_Mut2_GAAA_5mM_150mMK_1'][mask] = np.nan
    entire_lib_selected['dGerr_Mut2_GAAA_5mM_150mMK_1'][mask] = np.nan
    print(sum(mask))
    
    mask = entire_lib_selected['dGerr_Mut2_GUAA_1'] > error_threshold
    entire_lib_selected['dG_Mut2_GUAA_1'][mask] = np.nan
    entire_lib_selected['dGerr_Mut2_GUAA_1'][mask] = np.nan
    print(sum(mask))
#%%
#unique_receptors = sorted(list(set(list(entire_lib_selected.new_name))))
unique_receptors = sorted(list(set(list(entire_lib_selected.r_seq))))
#% Start analyzing only five scaffolds, create dataframe with all data (5scaffolds * conditions)
# for eact TLR ina a single row.
scaffolds_five = ['13854','14007','14073','35311_A','35600']
#aa = canonical.copy()
#aa = aa.set_index('old_idx')
#aa = aa['dG_Mut2_GAAA']
#aa = aa.loc[scaffolds_five]
#aa = aa.sort_values()
#scaffolds_five = list(aa.index)
#%
data = entire_lib_selected
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
prep_data,original_nan = prep_data_for_clustering_ver3(data_50_scaffolds,
                                                       dG_threshold,dG_replace,nan_threshold,
                                                       num_neighbors=num_neighbors)
#%% CATALOGUING RECEPTORS: Create a receptor types dataframe with the type.
A = entire_lib_selected.drop_duplicates(subset = 'r_seq',keep = 'last').copy()
A = A.set_index('r_seq')
#%
new_index = []
for sequence in prep_data.index:
    for receptor in reversed(A.index):
        if sequence == receptor:
            new_index.append(A.loc[receptor]['new_name'])
old_index = prep_data.index            
prep_data.index = new_index
prep_data['old_index'] = old_index   
#%
sublib5['new_name'] = sublib5.index + '_' +  sublib5['r_seq']
#%
all_receptors = new_index
receptor_types_df = pd.DataFrame(index = new_index)
receptor_types_df['type'] = 'uncatalogued'
receptor_types_df['r_seq'] = old_index
#%
for next_receptor in receptor_types_df.index:
    if next_receptor in list(sublib5['new_name']):
        receptor_types_df.loc[next_receptor]['type'] = 'other'
    else:
        if '11ntR' in next_receptor:
            receptor_types_df.loc[next_receptor]['type'] = '11ntR' 
            
        if 'IC3' in next_receptor:
            receptor_types_df.loc[next_receptor]['type'] = 'IC3' 
    
        if 'VC2' in next_receptor:
            receptor_types_df.loc[next_receptor]['type'] = 'VC2'
    
        if ('C7.' in next_receptor) | ('B7.' in next_receptor)| ('GAAC_receptor' in next_receptor) :
            receptor_types_df.loc[next_receptor]['type'] = 'in_vitro' 

#make sure that the WT sequences are in the correct category
receptor_types_df['type'][receptor_types_df['r_seq'] == 'UAUGG_CCUAAG'] = '11ntR'            
receptor_types_df['type'][receptor_types_df['r_seq'] == 'GAGGG_CCCUAAC'] = 'IC3'
receptor_types_df['type'][receptor_types_df['r_seq'] == 'GUAGG_CCUAAAC'] = 'VC2'
receptor_types_df['type'][receptor_types_df['r_seq'] == 'GAUGG_CCUGUAC'] = 'in_vitro'
receptor_types_df['type'][receptor_types_df['r_seq'] == 'UAUGG_CCUGUG'] = 'in_vitro'
receptor_types_df['type'][receptor_types_df['r_seq'] == 'GAAGGG_CCCCACGC'] = 'in_vitro'
                       
receptor_counts = receptor_types_df['type'].value_counts()
print(receptor_counts)
receptor_counts = receptor_counts.drop(['uncatalogued'])
color_list = ['blue','red','yellow','black','green']
plt.pie(receptor_counts,colors=color_list,startangle = 40,wedgeprops={'linewidth': 1,'edgecolor':'black', 'linestyle': 'solid'})
plt.axis('equal')
#% Save uncatalogued receptors
uncatalogued_receptors = receptor_types_df[receptor_types_df['type'] == 'uncatalogued']
uncatalogued_receptors.to_csv('uncatalogued_receptors.csv')
#% Drop 'uncatalogued' receptors from dataframe to be plotted and analyzed
# These uncatalogued receptors contain tandem bp receptors, and random crap that I got
# from the sequence database of group I introns
uncatalogued_receptors = receptor_types_df[receptor_types_df.type == 'uncatalogued']
prep_data = prep_data.drop(uncatalogued_receptors.index)
#%Return prep data to oringinal index
prep_data = prep_data.set_index('old_index')
#%
tandem_receptor = []
for receptor in uncatalogued_receptors.index:
    if ('tandem' in receptor) or ('T4' in receptor):
        tandem_receptor.append(receptor)
#%REMEMBER THAT THE NAME IS REVERSED WITH RESPECT TO THE SEQUENCE
control_CC_GG = entire_lib_selected[entire_lib_selected.r_seq == 'GG_CC']    
#Calculate the medians with respect to CHIP SCAFFOLD length
A = control_CC_GG.groupby(by = 'length')
#%
grouped_lib = entire_lib_selected.groupby('r_seq')
IC3 = entire_lib_selected[entire_lib_selected.r_seq == 'GAGGG_CCCUAAC' ]
VC2 = entire_lib_selected[entire_lib_selected.r_seq ==  'GUAGG_CCUAAAC']
IC3_alt = entire_lib_selected[entire_lib_selected.r_seq == 'GAGGA_UUCUAAC' ]

scaff = list(set(entire_lib_selected.old_idx))
#IC3 = entire_lib_selected[entire_lib_selected.new_name == 'IC3 _GAGGG_CCCUAAC_tertcontacts_0']
IC3 = IC3.set_index('old_idx')
IC3 = IC3.reindex(scaff)

#VC2 = entire_lib_selected[entire_lib_selected.new_name == 'VC2_GUAGG_CCUAAAC_tertcontacts_0']
VC2 = VC2.set_index('old_idx')
VC2 = VC2.reindex(scaff)
plt.scatter(IC3.dG_Mut2_GUAA_1,VC2.dG_Mut2_GUAA_1)
plt.plot([-14,6],[-14,6],'--k')
plt.xlim(-14,-6)
plt.ylim(-14,-6)
plt.title('GUAA binding IC3 vs VC2')


#IC3_alt = entire_lib_selected[entire_lib_selected.new_name == 'IC3_1U_2U_12A_GAGGA_UUCUAAC_tertcontacts_0']
IC3_alt = IC3_alt.set_index('old_idx')
IC3_alt = IC3_alt.reindex(scaff)

plt.figure()
plt.scatter(IC3.dG_Mut2_GUAA_1,IC3_alt.dG_Mut2_GUAA_1)
plt.plot([-14,6],[-14,6],'--k')
plt.xlim(-14,-6)
plt.ylim(-14,-6)
plt.title('GUAA binding IC3 vs IC3_alt')

plt.figure()
plt.scatter(IC3.dG_Mut2_GAAA,VC2.dG_Mut2_GAAA)
plt.plot([-14,6],[-14,6],'--k')
plt.xlim(-14,-6)
plt.ylim(-14,-6)
plt.title('GAAA binding IC3 vs VC2')

plt.figure()
plt.scatter(IC3.dG_Mut2_GAAA,IC3_alt.dG_Mut2_GAAA)
plt.plot([-14,6],[-14,6],'--k')
plt.xlim(-14,-6)
plt.ylim(-14,-6)
plt.title('GAAA binding IC3 vs IC3_alt')
#% Also drop uncatalogued receptors from the receptor_types dataframe
receptor_types_df = receptor_types_df.drop(uncatalogued_receptors.index)
#%% reinsert nan values VERY IMPORTANT!!!!!
# Heatmaps and many of the other plots need not to included nan values
# nans were assigned values by interpolation only for clustering and other 
# statistical tools that do not support missing values.
prep_data_with_nan = prep_data.copy()
prep_data_with_nan[original_nan] = np.nan
prep_data_with_nan.to_pickle('./prep_data_all_TLRs_5_scaffolds_with_nan.pkl')
#Because we are more concerned with the behavior of the TLR as the conditions
#and the scaffolds are changed, we substract the average of the entire row.
#Normalized each tetraloop-receptor with respect to its mean accross columns
mean_per_row = prep_data.mean(axis=1)
prep_data_norm = prep_data.copy()
prep_data_norm = prep_data_norm.sub(mean_per_row,axis=0)
prep_data_norm_with_nan = prep_data_norm.copy()
prep_data_norm_with_nan[original_nan] = np.nan
#%% Get information about the receptors; sublibrary, sequence, etc
#Create receptor_info dataframe that in combination with receptor_types_df
#should contain all the information I need about the receptors. 
r_seq = []
r_name = []
sublib_list = []
column_names = ['r_seq','r_name','new_name','sublibrary']
receptor_info = entire_lib_selected.drop_duplicates(['new_name'])
receptor_info = receptor_info[column_names]
receptor_info = receptor_info.set_index('new_name')
receptor_info = receptor_info.reindex(receptor_types_df.index)
#%% Create a single dataframe with all the information
receptor_info['type'] = receptor_types_df['type']
receptor_info['new_name'] = receptor_info.index
receptor_info = receptor_info.set_index('r_seq')
#%%
receptor_info.to_pickle('./receptor_info.pkl')
receptor_info.to_pickle('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/'
                        'initial_clustering_all_data/receptor_info.pkl')
#%%Create a matrix with receptor types for later plotting aside with the clustermap
#to show where the different receptors of different types are clusteringg
columns = ['type_11ntR','type_IC3','type_VC2','type_inVitro','type_other',]
receptors_types_matrix = pd.DataFrame(0,index=prep_data.index,columns=columns)
receptors_types_matrix.type_11ntR[receptor_info.type == '11ntR'] = 1
receptors_types_matrix.type_IC3[receptor_info.type == 'IC3'] = 1
receptors_types_matrix.type_VC2[receptor_info.type == 'VC2'] = 1
receptors_types_matrix.type_inVitro[receptor_info.type == 'in_vitro'] = 1
receptors_types_matrix.type_other[receptor_info.type == 'other'] = 1

receptors_types_matrix.to_pickle('./receptors_types_matrix.pkl')
receptors_types_matrix.to_pickle('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/'
                        'initial_clustering_all_data/receptors_types_matrix.pkl')
#%%INITITAL CLUSTERIN OF THE DATA; THIS IS WITHOUT DOING PCA ANALYSIS OR MUCH
#PREPROCESSING
#Clustermap without PCA 
z_withoutPCA = sch.linkage(prep_data_norm,method = 'ward')
cg_without_PCA = sns.clustermap(prep_data_norm_with_nan,row_linkage=z_withoutPCA, col_cluster=False,
                        vmin=-5,vmax=5,cmap='coolwarm')
cg_without_PCA = sns.clustermap(receptors_types_matrix,row_linkage=z_withoutPCA, col_cluster=False,
                        vmin=0,vmax=1,cmap='Greys')

#%%SCALING DATA DID NOT MAKE NOTICEABLE DIFFERENCE
#scaler = StandardScaler()
#df_test = scaler.fit_transform(prep_data_norm)
#new = pd.DataFrame(df_test,index = prep_data_norm.index)
#pca,transformed,loadings = doPCA(new)
#num_PCA = 9
#pd.Series(pca.explained_variance_ratio_).plot(kind='bar')
#plt.ylabel('fraction of variance \nexplained by each PC', fontsize=14)
#plt.tight_layout()
#print('Fraction explained by the first ',str(num_PCA), 'PCAs :',sum(pca.explained_variance_ratio_[:num_PCA]))
##
#plt.figure()
#cumulative = 1 - np.cumsum(pca.explained_variance_ratio_)
#plt.plot(range(len(cumulative)),cumulative)
#list_PCAs = list(transformed.columns[:num_PCA])
#z_pca = sch.linkage(transformed.loc[:,list_PCAs],method='ward',optimal_ordering=True) 
#cg_pca = sns.clustermap(prep_data_norm_with_nan,row_linkage=z_pca, col_cluster=False
#                        ,row_cluster=True, vmin=-3.2,vmax=3.2,cmap='coolwarm')
##
#X = transformed.loc[:,list_PCAs]
#c, coph_dists = cophenet(z_pca, pdist(X))
#print('cophenetic distance: ',c)
##
#reordered_data = pd.DataFrame(index = range(0,len(prep_data_norm)))
#reordered_data['name'] = list(prep_data_norm.index)
#reordered_data = reordered_data.reindex(list(cg_pca.dendrogram_row.reordered_ind))
#reordered_data.index = range(0,len(prep_data_norm))
##
#a= list(cg_pca.dendrogram_row.reordered_ind)
##Display Dendrogram a locate specific tetraloop-receptors
#distance_threshold = 12
#dendro = sch.dendrogram(z_pca,color_threshold=distance_threshold)
#plt.show()
#max_d = distance_threshold
#clusters = fcluster(z_pca, max_d, criterion='distance')
#number_clusters = max(clusters)
#print('number of clusters: ' + str(number_clusters))
##
##
#clusters_df = pd.DataFrame(clusters,index = prep_data_norm_with_nan.index,columns = ['cluster_No'])
#print('number of clusters based on distance of ',str(max_d), ':', str(number_clusters))
#WT_cluster = clusters_df.loc['11ntR_UAUGG_CCUAAG_tertcontacts_0'].cluster_No
#print('11ntR clustered with group ' + str(WT_cluster))
#C_WT = clusters_df[clusters_df.cluster_No == WT_cluster]
#print('in location: ' + str(reordered_data[reordered_data.name == '11ntR_UAUGG_CCUAAG_tertcontacts_0'].index))
#print('which has shape ' + str(C_WT.shape))
#
#IC3_cluster = clusters_df.loc['IC3 _GAGGG_CCCUAAC_tertcontacts_0'].cluster_No
#print('IC3 clustered with group ' + str(IC3_cluster))
#C_IC3 = clusters_df[clusters_df.cluster_No == IC3_cluster]
#print('in location: ' + str(reordered_data[reordered_data.name == 'IC3 _GAGGG_CCCUAAC_tertcontacts_0'].index))
#print('which has shape ' + str(C_IC3.shape))
#
#C72_cluster = clusters_df.loc['C7.2_GAUGG_CCUGUAC_tertcontacts_0'].cluster_No
#print('C7.2 clustered with group ' + str(C72_cluster))
#C_C72 = clusters_df[clusters_df.cluster_No == C72_cluster]
#print('in location: ' + str(reordered_data[reordered_data.name == 'C7.2_GAUGG_CCUGUAC_tertcontacts_0'].index))
#print('which has shape ' + str(C_C72.shape))
#
#C72_cluster = clusters_df.loc['IC3 _GAGGG_CCUUAUC_tertcontacts_4'].cluster_No
#print('max specificity for GUAAA clustered with group ' + str(C72_cluster))
#C_C72 = clusters_df[clusters_df.cluster_No == C72_cluster]
#print('which has shape ' + str(C_C72.shape))
#
#Vc2_cluster = clusters_df.loc['VC2_GUAGG_CCUAAAC_tertcontacts_0'].cluster_No
#print('Vc2 which showed weak specificity clustered with group ' + str(Vc2_cluster))
#C_Vc2 = clusters_df[clusters_df.cluster_No == Vc2_cluster]
#print('in location: ' + str(reordered_data[reordered_data.name == 'VC2_GUAGG_CCUAAAC_tertcontacts_0'].index))
#print('which has shape ' + str(C_Vc2.shape))

#%%CLUSTER THE DATA AGAIN BUT NOW WITH PCA ANALYSIS PRECEEDING THE CLUSTERING;
#FOR THE PCA, I AM USING THE FUNCTION THAT SARAH GAVE ME
'''------------PCA Analysis of raw data-----------------'''

#How many PCs shall we use????
num_PCA = 9
pca,transformed,loadings = doPCA(prep_data_norm)
#plot explained variance by PC
pd.Series(pca.explained_variance_ratio_).plot(kind='bar')
plt.ylabel('fraction of variance \nexplained by each PC', fontsize=14)
plt.tight_layout()
print('Fraction explained by the first ',str(num_PCA), 'PCAs :',sum(pca.explained_variance_ratio_[:num_PCA]))

plt.figure()
cumulative = 1 - np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(len(cumulative)),cumulative)

list_PCAs = list(transformed.columns[:num_PCA])
#%%
z_pca = sch.linkage(transformed.loc[:,list_PCAs],method='ward',optimal_ordering=True) 
cg_pca = sns.clustermap(prep_data_norm_with_nan,row_linkage=z_pca, col_cluster=False
                        ,row_cluster=True, vmin=-3.2,vmax=3.2,cmap='coolwarm')
#cg_pca.savefig('/Volumes/NO NAME/Clustermaps/five_scaff_02_10_2018.svg')


cg_receptors= sns.clustermap(receptors_types_matrix,row_linkage=z_pca, col_cluster=False,
                        vmin=0,vmax=1,cmap='Greys')
#cg_receptors.savefig('/Volumes/NO NAME/Clustermaps/clustermap_receptors.svg')

cg_mean_dG= sns.clustermap(mean_per_row,row_linkage=z_pca, col_cluster=False,
                        vmin=-10,vmax=-7,cmap='Blues_r')
#cg_mean_dG.savefig('/Volumes/NO NAME/Clustermaps/clustermap_mean_dG.svg')
X = transformed.loc[:,list_PCAs]
c, coph_dists = cophenet(z_pca, pdist(X))
print('cophenetic distance: ',c)
#%%
columns = ['type_11ntR','type_IC3','type_VC2','type_inVitro']
wt_types_matrix = pd.DataFrame(0,index=prep_data.index,columns=columns)
wt_types_matrix.type_11ntR[receptor_info.index == 'UAUGG_CCUAAG'] = 1
wt_types_matrix.type_IC3[receptor_info.index == 'GAGGG_CCCUAAC'] = 1
wt_types_matrix.type_VC2[receptor_info.index == 'GUAGG_CCUAAAC'] = 1
wt_types_matrix.type_inVitro[receptor_info.index == 'GAUGG_CCUGUAC'] = 1
wt_types_matrix.type_inVitro[receptor_info.index == 'UAUGG_CCUGUG'] = 1
wt_types_matrix.type_inVitro[receptor_info.index == 'GAAGGG_CCCCACGC'] = 1
#wt_types_matrix.type_other[receptor_info.type == 'other'] = 1

cg_receptors= sns.clustermap(wt_types_matrix,row_linkage=z_pca, col_cluster=False,
                        vmin=0,vmax=1,cmap='Blues')
#cg_receptors.savefig('/Volumes/NO NAME/Clustermaps/clustermap_receptors_WT.svg')
#%%
#Create heatmap to place next to ddG clustermap in which we show which receptors 
#show up in Nature and which ones do not.

#import natural frequency data for 11ntR variants 
new_data_path = '/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/January_2019/'
details_11ntR = pd.read_csv(new_data_path + 'details_11ntR_updated_01_09_2019.csv' )

#select only the ones that were found to have a GNRA receptor 
details_11ntR = details_11ntR[details_11ntR['GNRA_found'] == 1]
print ('original total number of 11ntR variants instances found: ' + str(len(details_11ntR)))
#select only the ones that have their secondary structure confirmed 
details_11ntR = details_11ntR[details_11ntR['confirmed'] == 1]
print('after visual confirmation of secondary structures: ' + str(len(details_11ntR)))
#
natural_receptors_11ntR = list(set(details_11ntR['receptor']))

natural_11ntR_matrix = pd.DataFrame(index = receptors_types_matrix.index)
found_11ntR = []
for sequence in natural_11ntR_matrix.index:
    if sequence in natural_receptors_11ntR:
        found_11ntR.append(1)
    else:
        found_11ntR.append(0)
natural_11ntR_matrix['found'] = found_11ntR

cg_11ntR_natural= sns.clustermap(natural_11ntR_matrix,row_linkage=z_pca, col_cluster=False,
                        vmin=0,vmax=1,cmap='Greys')

cg_11ntR_natural.savefig(new_data_path + 'all_TLRs_clust_11ntR_natural_.svg')

#----------------------------------------------------
#import natural frequency data for IC3 variants 

details_IC3 = pd.read_csv(new_data_path + 'details_IC3_updated_01_10_2019.csv' )

#select only the ones that were found to have a GNRA receptor 
details_IC3 = details_IC3[details_IC3['GNRA_found'] == 1]
print ('original total number of IC3 variants instances found: ' + str(len(details_IC3)))
#select only the ones that have their secondary structure confirmed 
details_IC3 = details_IC3[details_IC3['confirmed'] == 1]
print('after visual confirmation of secondary structures: ' + str(len(details_IC3)))
#
natural_receptors_IC3 = list(set(details_IC3['receptor']))

natural_IC3_matrix = pd.DataFrame(index = receptors_types_matrix.index)
found_IC3 = []
for sequence in natural_IC3_matrix.index:
    if sequence in natural_receptors_IC3:
        found_IC3.append(1)
    else:
        found_IC3.append(0)
natural_IC3_matrix['found'] = found_IC3

cg_IC3_natural= sns.clustermap(natural_IC3_matrix,row_linkage=z_pca, col_cluster=False,
                        vmin=0,vmax=1,cmap='Greys')
cg_IC3_natural.savefig(new_data_path + 'all_TLRs_clust_IC3_natural_.svg')

#--------------------------------------------------------
#import natural frequency data for VC2 variants 

details_VC2 = pd.read_csv(new_data_path + 'details_VC2_updated_01_10_2019.csv' )

#select only the ones that were found to have a GNRA receptor 
details_VC2 = details_VC2[details_VC2['GNRA_found'] == 1]
print ('original total number of VC2 variants instances found: ' + str(len(details_VC2)))
#select only the ones that have their secondary structure confirmed 
details_VC2 = details_VC2[details_VC2['confirmed'] == 1]
print('after visual confirmation of secondary structures: ' + str(len(details_VC2)))
#
natural_receptors_VC2 = list(set(details_VC2['receptor']))

natural_VC2_matrix = pd.DataFrame(index = receptors_types_matrix.index)
found_VC2 = []
for sequence in natural_VC2_matrix.index:
    if sequence in natural_receptors_VC2:
        found_VC2.append(1)
    else:
        found_VC2.append(0)
natural_VC2_matrix['found'] = found_VC2

cg_VC2_natural= sns.clustermap(natural_VC2_matrix,row_linkage=z_pca, col_cluster=False,
                        vmin=0,vmax=1,cmap='Greys')
cg_VC2_natural.savefig(new_data_path + 'all_TLRs_clust_VC2_natural_.svg')
#%%
reordered_data = pd.DataFrame(index = range(0,len(prep_data_norm)))
reordered_data['name'] = list(prep_data_norm.index)
reordered_data = reordered_data.reindex(list(cg_pca.dendrogram_row.reordered_ind))
reordered_data.index = range(0,len(prep_data_norm))
#%%
a= list(cg_pca.dendrogram_row.reordered_ind)
#%%Display Dendrogram a locate specific tetraloop-receptors
distance_threshold = 12
dendro = sch.dendrogram(z_pca,color_threshold=distance_threshold)
plt.show()
max_d = distance_threshold
clusters = fcluster(z_pca, max_d, criterion='distance')
number_clusters = max(clusters)
print('number of clusters: ' + str(number_clusters))
#%%
clusters_df = pd.DataFrame(clusters,index = prep_data_norm_with_nan.index,columns = ['cluster_No'])
print('number of clusters based on distance of ',str(max_d), ':', str(number_clusters))
WT_cluster = clusters_df.loc['UAUGG_CCUAAG'].cluster_No
print('11ntR clustered with group ' + str(WT_cluster))
C_WT = clusters_df[clusters_df.cluster_No == WT_cluster]
print('in location: ' + str(reordered_data[reordered_data.name == 'UAUGG_CCUAAG'].index))
print('which has shape ' + str(C_WT.shape))

IC3_cluster = clusters_df.loc['GAGGG_CCCUAAC'].cluster_No
print('IC3 clustered with group ' + str(IC3_cluster))
C_IC3 = clusters_df[clusters_df.cluster_No == IC3_cluster]
print('in location: ' + str(reordered_data[reordered_data.name == 'GAGGG_CCCUAAC'].index))
print('which has shape ' + str(C_IC3.shape))

C72_cluster = clusters_df.loc['GAUGG_CCUGUAC'].cluster_No
print('C7.2 clustered with group ' + str(C72_cluster))
C_C72 = clusters_df[clusters_df.cluster_No == C72_cluster]
print('in location: ' + str(reordered_data[reordered_data.name == 'GAUGG_CCUGUAC'].index))
print('which has shape ' + str(C_C72.shape))

C72_cluster = clusters_df.loc['GAGGG_CCUUAUC'].cluster_No
print('max specificity for GUAAA clustered with group ' + str(C72_cluster))
C_C72 = clusters_df[clusters_df.cluster_No == C72_cluster]
print('which has shape ' + str(C_C72.shape))

Vc2_cluster = clusters_df.loc['GUAGG_CCUAAAC'].cluster_No
print('Vc2 which showed weak specificity clustered with group ' + str(Vc2_cluster))
C_Vc2 = clusters_df[clusters_df.cluster_No == Vc2_cluster]
print('in location: ' + str(reordered_data[reordered_data.name == 'GUAGG_CCUAAAC'].index))
print('which has shape ' + str(C_Vc2.shape))

C79_cluster = clusters_df.loc['UAUGG_CCUGAAG'].cluster_No
print('C79  clustered with group ' + str(C79_cluster))
C_C79 = clusters_df[clusters_df.cluster_No == C79_cluster]
print('in location: ' + str(reordered_data[reordered_data.name == 'UAUGG_CCUGAAG'].index))
print('which has shape ' + str(C_C79.shape))

C734_cluster = clusters_df.loc['GAAGGG_CCCCACGC' ].cluster_No
print('C7.34  clustered with group ' + str(C734_cluster))
C_C734 = clusters_df[clusters_df.cluster_No == C734_cluster]
print('in location: ' + str(reordered_data[reordered_data.name == 'GAAGGG_CCCCACGC' ].index))
print('which has shape ' + str(C_C734.shape))

C743_cluster = clusters_df.loc['AAUGG_CCUGCC'].cluster_No
print('C7.43  clustered with group ' + str(C743_cluster))
C_C743 = clusters_df[clusters_df.cluster_No == C743_cluster]
print('in location: ' + str(reordered_data[reordered_data.name == 'AAUGG_CCUGCC' ].index))
print('which has shape ' + str(C_C743.shape))
#%%

#receptor = 'C7.9_UAUGG_CCUGAAG_tertcontacts_0' 
#receptor = 'C7.43_AAUGG_CCUGCC_tertcontacts_0' 
#receptor = 'C7.22_GAUGG_CCUGCAC_tertcontacts_0' 
#receptor = 'C7.34_GAAGGG_CCCCACGC_tertcontacts_0' 
receptor = 'C7.10_UAUGG_CCUGUG_tertcontacts_0' 
#%% plot profiles with respect to 11ntR_WT
#FOR 5 conditions
low_lim = -14
high_lim = -4
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]

columns_to_plot = prep_data.columns
WT_data = prep_data_with_nan.loc['11ntR_UAUGG_CCUAAG_tertcontacts_0']
receptors = 'C7.2_GAUGG_CCUGUAC_tertcontacts_0'
plt.figure()
#plot 30mM 
plt.scatter(WT_data[columns_to_plot[0:5]],prep_data_with_nan.loc[receptors][columns_to_plot[0:5]],s=120,edgecolors='k',c='orange')
#plot 5 mM GAAA
plt.scatter(WT_data[columns_to_plot[5:10]],prep_data_with_nan.loc[receptors][columns_to_plot[5:10]],s=120,edgecolors='k',c='orange',marker='s')
#plot 5 mM + 150K
plt.scatter(WT_data[columns_to_plot[10:15]],prep_data_with_nan.loc[receptors][columns_to_plot[10:15]],s=120,edgecolors='k',c='orange',marker ='*')
#plot GUAA
plt.scatter(WT_data[columns_to_plot[15:20]],prep_data_with_nan.loc[receptors][columns_to_plot[15:20]],s=120,edgecolors='k',c='orange',marker ='^')

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

######################################
WT_data = prep_data_with_nan.loc['11ntR_UAUGG_CCUAAG_tertcontacts_0']
receptors = 'IC3 _GAGGG_CCUUAUC_tertcontacts_4'
plt.figure()
#plot 30mM 
plt.scatter(WT_data[columns_to_plot[0:5]],prep_data_with_nan.loc[receptors][columns_to_plot[0:5]],s=120,edgecolors='k',c='orange')
#plot 5 mM GAAA
plt.scatter(WT_data[columns_to_plot[5:10]],prep_data_with_nan.loc[receptors][columns_to_plot[5:10]],s=120,edgecolors='k',c='orange',marker='s')
#plot 5 mM + 150K
plt.scatter(WT_data[columns_to_plot[10:15]],prep_data_with_nan.loc[receptors][columns_to_plot[10:15]],s=120,edgecolors='k',c='orange',marker ='*')
#plot GUAA
plt.scatter(WT_data[columns_to_plot[15:20]],prep_data_with_nan.loc[receptors][columns_to_plot[15:20]],s=120,edgecolors='k',c='orange',marker ='^')

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

###########################
WT_data = prep_data_with_nan.loc['11ntR_UAUGG_CCUAAG_tertcontacts_0']
receptors = 'VC2_GUAGG_CCUAAAC_tertcontacts_0'
plt.figure()
#plot 30mM 
plt.scatter(WT_data[columns_to_plot[0:5]],prep_data_with_nan.loc[receptors][columns_to_plot[0:5]],s=120,edgecolors='k',c='orange')
#plot 5 mM GAAA
plt.scatter(WT_data[columns_to_plot[5:10]],prep_data_with_nan.loc[receptors][columns_to_plot[5:10]],s=120,edgecolors='k',c='orange',marker='s')
#plot 5 mM + 150K
plt.scatter(WT_data[columns_to_plot[10:15]],prep_data_with_nan.loc[receptors][columns_to_plot[10:15]],s=120,edgecolors='k',c='orange',marker ='*')
#plot GUAA
plt.scatter(WT_data[columns_to_plot[15:20]],prep_data_with_nan.loc[receptors][columns_to_plot[15:20]],s=120,edgecolors='k',c='orange',marker ='^')

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



###########################
WT_data = prep_data_with_nan.loc['11ntR_UAUGG_CCUAAG_tertcontacts_0']
receptors = 'C7.9_UAUGG_CCUGAAG_tertcontacts_0'
plt.figure()
#plot 30mM 
plt.scatter(WT_data[columns_to_plot[0:5]],prep_data_with_nan.loc[receptors][columns_to_plot[0:5]],s=120,edgecolors='k',c='orange')
#plot 5 mM GAAA
plt.scatter(WT_data[columns_to_plot[5:10]],prep_data_with_nan.loc[receptors][columns_to_plot[5:10]],s=120,edgecolors='k',c='orange',marker='s')
#plot 5 mM + 150K
plt.scatter(WT_data[columns_to_plot[10:15]],prep_data_with_nan.loc[receptors][columns_to_plot[10:15]],s=120,edgecolors='k',c='orange',marker ='*')
#plot GUAA
plt.scatter(WT_data[columns_to_plot[15:20]],prep_data_with_nan.loc[receptors][columns_to_plot[15:20]],s=120,edgecolors='k',c='orange',marker ='^')

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
#%% plot profiles with respect to 11ntR_WT
#FOR 50 conditions
low_lim = -14
high_lim = -6
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]

#receptor = 'VC2_GUAGG_CCUAAAC_tertcontacts_0'
#receptor = 'C7.2_GAUGG_CCUGUAC_tertcontacts_0'
#receptor = 'IC3 _GAGGG_CCCUAAC_tertcontacts_0'
#receptor = '11ntR_9C_UACGG_CCUAAG_tertcontacts_0'
#receptor = 'C7.9_UAUGG_CCUGAAG_tertcontacts_0' 
#receptor = 'C7.43_AAUGG_CCUGCC_tertcontacts_0' 
#receptor = 'C7.22_GAUGG_CCUGCAC_tertcontacts_0' 
#receptor = 'GAAGGG_CCCCACGC'#C7.34 
#receptor = 'UAUGG_CCUGUG' #C7.10
receptor = 'GAGGG_CCUUAUC' #max specificity for GUAA

color_to_plot = 'blue'


library_to_plot = entire_lib_selected.copy().groupby('r_seq')
to_plot_x = library_to_plot.get_group('UAUGG_CCUAAG').set_index('old_idx')
to_plot_x = to_plot_x.reindex(scaff)
to_plot_y = library_to_plot.get_group(receptor).set_index('old_idx')
to_plot_y = to_plot_y.reindex(scaff)


plt.figure()
#plot 30mM 
plt.scatter(to_plot_x['dG_Mut2_GAAA'],to_plot_y['dG_Mut2_GAAA'],s=120,edgecolors='k',c=color_to_plot)
#plot 30 mM GUAA
plt.scatter(to_plot_x['dG_Mut2_GUAA_1'],to_plot_y['dG_Mut2_GUAA_1'],s=120,edgecolors='k',c=color_to_plot,marker='s')
#plot 5 mM + 150K
plt.scatter(to_plot_x['dG_Mut2_GAAA_5mM_150mMK_1'],to_plot_y['dG_Mut2_GAAA_5mM_150mMK_1'],s=150,edgecolors='k',c= color_to_plot,marker ='*')
#plot 5mM
plt.scatter(to_plot_x['dG_Mut2_GAAA_5mM_2'],to_plot_y['dG_Mut2_GAAA_5mM_2'],s=120,edgecolors='k',c= color_to_plot,marker ='^')

plt.plot(x,x,':k', linewidth = 2.5)
plt.plot(x,y_thres,'--k',linewidth = 1.5)
plt.plot(y_thres,x,'--k',linewidth = 1.5)
plt.xlim(low_lim,high_lim)
plt.ylim(low_lim,high_lim)
plt.xticks(list(range(-14,-4,2)))
plt.yticks(list(range(-14,-4,2)))
plt.tick_params(axis='both', which='major', labelsize=24)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta$G$^{11ntR}_{bind}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta$G$^{mut}_{bind}$ (kcal/mol)',fontsize=22)
#%%THIS IS REDUNDANT, BUT I'LL LEAVE IT ANYWAY
receptors_analyzed = list(prep_data_with_nan.index)
lib_analyzed = entire_lib_selected[entire_lib_selected['new_name'].isin(receptors_analyzed)]
mylist =[]
for receptors in receptors_analyzed:
    if ('tertcontacts_0' in receptors) and ('IC3' in receptors):
        mylist = mylist + [receptors]
#%%A LOT OF THE ANALYSIS BELOW WAS JUST FOR VERIFICATION AND DOES NOT NEED TO 
#BE RUN OR INCLUDED
#As a comparison I am using the PCA package instead of Sarah's function
#but is basically the same thing.  So everything is looking good. 
#for comparison use 10 PCs and compare to other PCs
pca = PCA(n_components = 9)
pca.fit(prep_data_norm)
transformed_data = pca.transform(prep_data_norm)
transformed_data_df = pd.DataFrame(transformed_data, index = prep_data_norm.index)
#transformed_data_df = transformed_data_df/transformed_data_df.std()
z_new_pca = sch.linkage(transformed_data_df,method='ward') 
cg_pca_new = sns.clustermap(prep_data_norm_with_nan,row_linkage=z_new_pca, col_cluster=False
                        , vmin=-3.2,vmax=3.2,cmap='coolwarm')
cg_receptors_new= sns.clustermap(receptors_types_matrix,row_linkage=z_new_pca, col_cluster=False,
                        vmin=0,vmax=1,cmap='Greys')
distance_threshold = 12
#sch.dendrogram(z_pca,color_threshold=distance_threshold)
#plt.show()
max_d = distance_threshold
clusters = fcluster(z_new_pca, max_d, criterion='distance')
number_clusters = max(clusters)
clusters_df = pd.DataFrame(clusters,index = prep_data_norm_with_nan.index,columns = ['cluster_No'])
print('number of clusters based on distance of ',str(max_d), ':', str(number_clusters))
WT_cluster = clusters_df.loc['11ntR_UAUGG_CCUAAG_tertcontacts_0'].cluster_No
print('11ntR clustered with group ' + str(WT_cluster))
C_WT = clusters_df[clusters_df.cluster_No == WT_cluster]
print('which has shape ' + str(C_WT.shape))

C72_cluster = clusters_df.loc['C7.2_GAUGG_CCUGUAC_tertcontacts_0'].cluster_No
print('C7.2 clustered with group ' + str(C72_cluster))
C_C72 = clusters_df[clusters_df.cluster_No == C72_cluster]
print('which has shape ' + str(C_C72.shape))
comparison_cluster_assignment = clusters_df.copy()
#%% AGAIN, THIS MAY NOT BE NECESSARY
#rChange number of PCs taken cluster and compare to older.
pca = PCA(n_components = 10)
pca.fit(prep_data_norm)
transformed_data = pca.transform(prep_data_norm)
transformed_data_df = pd.DataFrame(transformed_data, index = prep_data_norm.index)
z_new_pca = sch.linkage(transformed_data_df,method='ward') 
distance_threshold = 12
#sch.dendrogram(z_pca,color_threshold=distance_threshold)
#plt.show()
max_d = distance_threshold
clusters = fcluster(z_new_pca, max_d, criterion='distance')
number_clusters = max(clusters)
clusters_df = pd.DataFrame(clusters,index = prep_data_norm_with_nan.index,columns = ['cluster_No'])
print('number of clusters based on distance of ',str(max_d), ':', str(number_clusters))
WT_cluster = clusters_df.loc['11ntR_UAUGG_CCUAAG_tertcontacts_0'].cluster_No
print('11ntR clustered with group ' + str(WT_cluster))
C_WT = clusters_df[clusters_df.cluster_No == WT_cluster]
print('which has shape ' + str(C_WT.shape))

C72_cluster = clusters_df.loc['C7.2_GAUGG_CCUGUAC_tertcontacts_0'].cluster_No
print('C7.2 clustered with group ' + str(C72_cluster))
C_C72 = clusters_df[clusters_df.cluster_No == C72_cluster]
print('which has shape ' + str(C_C72.shape))

new_cluster_assignment = clusters_df.copy()
new_cluster_assignment.columns = ['new_cluster']
new_cluster_assignment['cluster_with_10PCs_comparison'] = comparison_cluster_assignment['cluster_No']
#%%Create a row with limits that indicate TLRs that do not have sufficient 
#data and therefore not much can be said about them
#The calculation was done based only on columns at 30 mM Mg as they are 
#the ones that contain most of the information
#rows (TLRs) that had more than limits_thr of data above stability threshold
limits_thr = 0.50
#take subsection with data at 30 mM GAAA and 30 mM GUAA
high_salt_columns = list(prep_data.columns[0:5]) + list(prep_data.columns[15:21])
high_salt_data = prep_data[high_salt_columns]
mask = high_salt_data == -7.1
limits_per_row = mask.sum(axis = 1)
limits_thr_num = len(mask.columns) * limits_thr
above_limit = limits_per_row > limits_thr_num

#NOTICE THAT THIS IS USING z_pca, #WHICH WAS GENERATED MANY CELLS ABOVE
cg_limits= sns.clustermap(above_limit,row_linkage=z_pca, col_cluster=False,
                        vmin=0,vmax=1,cmap='Greys')
cg_limits.savefig('/Volumes/NO NAME/Clustermaps/limits.svg')

#%%
#separate data that is above limit
prep_data_norm_nan_limit = prep_data_norm_with_nan[above_limit]
prep_data_norm_limit = prep_data_norm[above_limit]

prep_data_norm_nan_above_limit = prep_data_norm_with_nan[~above_limit]
prep_data_norm_above_limit = prep_data_norm[~above_limit]

num_PCA = 9
pca,transformed,loadings = doPCA(prep_data_norm_above_limit)
#plot explained variance by PC
pd.Series(pca.explained_variance_ratio_).plot(kind='bar')
plt.ylabel('fraction of variance \nexplained by each PC', fontsize=14)
plt.tight_layout()
print('Fraction explained by the first ',str(num_PCA), 'PCAs :',sum(pca.explained_variance_ratio_[:num_PCA]))
list_PCAs = list(transformed.columns[:num_PCA])

z_pca = sch.linkage(transformed.loc[:,list_PCAs],method='ward',optimal_ordering=False) 
cg_pca = sns.clustermap(prep_data_norm_nan_above_limit,row_linkage=z_pca, col_cluster=False
                        , vmin=-3.2,vmax=3.2,cmap='coolwarm')
#cg_pca.savefig('/Volumes/NO NAME/Clustermaps/clustermap_alldata_ddG.svg')

#Display Dendrogram a locate specific tetraloop-receptors
plt.figure()
distance_threshold = 12
dendro = sch.dendrogram(z_pca,color_threshold=distance_threshold)
plt.show()
max_d = distance_threshold
clusters = fcluster(z_pca, max_d, criterion='distance')
number_clusters = max(clusters)
print('number of clusters: ' + str(number_clusters))

reordered_data = pd.DataFrame(index = range(0,len(prep_data_norm_nan_above_limit)))
reordered_data['name'] = list(prep_data_norm_nan_above_limit.index)
reordered_data = reordered_data.reindex(list(cg_pca.dendrogram_row.reordered_ind))
reordered_data.index = range(0,len(prep_data_norm_nan_above_limit))

clusters_df = pd.DataFrame(clusters,index = prep_data_norm_nan_above_limit.index,columns = ['cluster_No'])
print('number of clusters based on distance of ',str(max_d), ':', str(number_clusters))


WT_cluster = clusters_df.loc['11ntR_UAUGG_CCUAAG_tertcontacts_0'].cluster_No
print('11ntR clustered with group ' + str(WT_cluster))
C_WT = clusters_df[clusters_df.cluster_No == WT_cluster]
print('in location: ' + str(reordered_data[reordered_data.name == '11ntR_UAUGG_CCUAAG_tertcontacts_0'].index))
print('which has shape ' + str(C_WT.shape))


IC3_cluster = clusters_df.loc['IC3 _GAGGG_CCCUAAC_tertcontacts_0'].cluster_No
print('IC3 clustered with group ' + str(IC3_cluster))
C_IC3 = clusters_df[clusters_df.cluster_No == IC3_cluster]
print('in location: ' + str(reordered_data[reordered_data.name == 'IC3 _GAGGG_CCCUAAC_tertcontacts_0'].index))
print('which has shape ' + str(C_IC3.shape))

C72_cluster = clusters_df.loc['C7.2_GAUGG_CCUGUAC_tertcontacts_0'].cluster_No
print('C7.2 clustered with group ' + str(C72_cluster))
C_C72 = clusters_df[clusters_df.cluster_No == C72_cluster]
print('in location: ' + str(reordered_data[reordered_data.name == 'C7.2_GAUGG_CCUGUAC_tertcontacts_0'].index))
print('which has shape ' + str(C_C72.shape))

C72_cluster = clusters_df.loc['IC3 _GAGGG_CCUUAUC_tertcontacts_4'].cluster_No
print('max specificity for GUAAA clustered with group ' + str(C72_cluster))
C_C72 = clusters_df[clusters_df.cluster_No == C72_cluster]
print('which has shape ' + str(C_C72.shape))

Vc2_cluster = clusters_df.loc['VC2_GUAGG_CCUAAAC_tertcontacts_0'].cluster_No
print('Vc2 which showed weak specificity clustered with group ' + str(Vc2_cluster))
C_Vc2 = clusters_df[clusters_df.cluster_No == Vc2_cluster]
print('in location: ' + str(reordered_data[reordered_data.name == 'VC2_GUAGG_CCUAAAC_tertcontacts_0'].index))
print('which has shape ' + str(C_Vc2.shape))
#%%
cg_receptors= sns.clustermap(receptors_types_matrix,row_linkage=z_pca, col_cluster=False,
                        vmin=0,vmax=1,cmap='Greys')
#cg_receptors.savefig('/Volumes/NO NAME/Clustermaps/clustermap_receptors.svg')

cg_mean_dG= sns.clustermap(mean_per_row,row_linkage=z_pca, col_cluster=False,
                        vmin=-10,vmax=-7,cmap='Blues_r')
#cg_mean_dG.savefig('/Volumes/NO NAME/Clustermaps/clustermap_mean_dG.svg')
X = transformed.loc[:,list_PCAs]
c, coph_dists = cophenet(z_pca, pdist(X))
print('cophenetic distance: ',c)

#%% NOW WE ARE GOING TO WANT TO ALL DATA, HISTOGRAMS, SCATTER PLOTS ETC
#BUT ONLY WANT TO INCLUDE DATA THAT HAS BEEN ANALYZED ABOVE 
original_data = data_50_scaffolds.copy()
original_data = original_data.reindex(prep_data_with_nan.index)
#%%  HISTOGRAMS AND SCATTER PLOTS OF ALL DATA FOR 5 MAIN SCAFFOLDS
#stack the data for each scaffold into different salt conditions.
scaffolds_five = ['13854','14007','14073','35311_A','35600']
data_GAAA_30Mg_list = []
data_GAAA_5Mg_list = []
data_GAAA_5Mg150K_list = []
data_GUAA_30Mg_list = []
for scaffold in scaffolds_five:
    data_GAAA_30Mg_list = data_GAAA_30Mg_list + list(original_data['dG_30mM_Mg_GAAA_' + scaffold].values)
    data_GAAA_5Mg_list = data_GAAA_5Mg_list + list(original_data['dG_5mM_Mg_GAAA_' + scaffold].values)
    data_GAAA_5Mg150K_list = data_GAAA_5Mg150K_list + list(original_data['dG_5Mg150K_GAAA_' + scaffold].values)
    data_GUAA_30Mg_list = data_GUAA_30Mg_list + list(original_data['dG_30mM_Mg_GUAA_' + scaffold].values)

data_GAAA_30Mg_series = pd.Series(data_GAAA_30Mg_list)
data_GAAA_5Mg_series = pd.Series(data_GAAA_5Mg_list)
data_GAAA_5Mg150K_series = pd.Series(data_GAAA_5Mg150K_list)
data_GUAA_30Mg_series = pd.Series(data_GUAA_30Mg_list)
#Histograms of all data
R=[-14, -5] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
axs = axs.ravel()
axs[0].hist(data_GAAA_30Mg_series.dropna(),range=R,bins = b)
axs[1].hist(data_GUAA_30Mg_series.dropna(),range=R,bins = b)
axs[2].hist(data_GAAA_5Mg_series.dropna(),range=R,bins = b)
axs[3].hist(data_GAAA_5Mg150K_series.dropna(),range=R,bins = b)

# Scatter Plot of 30 mM vs. 5 mM
low_lim = -14
high_lim = -6
dG_threshold = -7.1
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
plt.figure()
x_data = data_GAAA_30Mg_series
y_data = data_GAAA_5Mg_series
plt.scatter(x_data,y_data)
#calculate ddG_avg for values above threshold
x_data_thr = x_data[x_data<dG_threshold]
y_data_thr = y_data[y_data<dG_threshold]
ddG = y_data_thr - x_data_thr
ddG_avg = ddG.mean()
plt.plot(x,y_thres,':k',linewidth = 0.5)
plt.plot(y_thres,x,':k',linewidth = 0.5)
plt.plot(x,x,':k')
plt.plot(x,[element+ddG_avg for element in x],'--r')
plt.xlim(low_lim,high_lim)
plt.ylim(low_lim,high_lim)
plt.axes().set_aspect('equal')

# Scatter Plot of 30 mM vs. 5 mM + potassium
low_lim = -14
high_lim = -6
dG_threshold = -7.1
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
plt.figure()
x_data = data_GAAA_30Mg_series
y_data = data_GAAA_5Mg150K_series
plt.scatter(x_data,y_data)
#calculate ddG_avg for values above threshold
x_data_thr = x_data[x_data<dG_threshold]
y_data_thr = y_data[y_data<dG_threshold]
ddG = y_data_thr - x_data_thr
ddG_avg = ddG.mean()
plt.plot(x,y_thres,':k',linewidth = 0.5)
plt.plot(y_thres,x,':k',linewidth = 0.5)
plt.plot(x,x,':k')
plt.plot(x,[element+ddG_avg for element in x],'--r')
plt.xlim(low_lim,high_lim)
plt.ylim(low_lim,high_lim)
plt.axes().set_aspect('equal')

#%% HISTOGRAMS AND SCATTERT PLOTS FOR ALL DATA FOR ALL SCAFFOLDS
data_GAAA_30Mg_series = lib_analyzed.dG_Mut2_GAAA
data_GAAA_5Mg_series = lib_analyzed.dG_Mut2_GAAA_5mM_2
data_GAAA_5Mg150K_series = lib_analyzed.dG_Mut2_GAAA_5mM_150mMK_1
data_GUAA_30Mg_series = lib_analyzed.dG_Mut2_GUAA_1
#Histograms of all data
R=[-14, -5] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
axs = axs.ravel()
axs[0].hist(data_GAAA_30Mg_series.dropna(),range=R,bins = b)
axs[1].hist(data_GUAA_30Mg_series.dropna(),range=R,bins = b)
axs[2].hist(data_GAAA_5Mg_series.dropna(),range=R,bins = b)
axs[3].hist(data_GAAA_5Mg150K_series.dropna(),range=R,bins = b)
#%% What are the averages for each condition across the entire thing 
#   WHEN TAKING INTO ACCOUNT THE VALUES ABOVE THE LIMIT BY MAKING THEM EQUAL TO 
#   LIMIT.
curated_data_30Mg_GAAA = data_GAAA_30Mg_series.copy()
curated_data_30Mg_GAAA[curated_data_30Mg_GAAA>dG_threshold] = dG_threshold
curated_data_30Mg_GAAA = curated_data_30Mg_GAAA .dropna()
print('The average for all data taken (within the threshold) at 30 mM MG for GAAA is : ' + str(curated_data_30Mg_GAAA.mean()))
print('The std for all data taken (within the threshold) at 30 mM MG for GAAA is : ' + str(curated_data_30Mg_GAAA.std()))
print('number of measurements :' + str(curated_data_30Mg_GAAA.size))

curated_data_30Mg_GUAA = data_GUAA_30Mg_series.copy()
curated_data_30Mg_GUAA[curated_data_30Mg_GUAA>dG_threshold] = dG_threshold
curated_data_30Mg_GUAA = curated_data_30Mg_GUAA.dropna()
curated_data_30Mg_GUAA[curated_data_30Mg_GUAA>dG_threshold] = dG_threshold
print('The average for all data taken (within the threshold) at 30 mM MG for GUAA is : ' + str(curated_data_30Mg_GUAA.mean()))
print('The std for all data taken (within the threshold) at 30 mM MG for GUAA is : ' + str(curated_data_30Mg_GUAA.std()))
print('number of measurements :' + str(curated_data_30Mg_GUAA.size))


curated_data_5Mg_GAAA = data_GAAA_5Mg_series.copy()
curated_data_5Mg_GAAA[curated_data_5Mg_GAAA>dG_threshold] = dG_threshold
curated_data_5Mg_GAAA = curated_data_5Mg_GAAA .dropna()
print('The average for all data taken (within the threshold) at 5 mM MG for GAAA is : ' + str(curated_data_5Mg_GAAA.mean()))
print('The std for all data taken (within the threshold) at 5 mM MG for GAAA is : ' + str(curated_data_5Mg_GAAA.std()))
print('number of measurements :' + str(curated_data_5Mg_GAAA.size))

curated_data_5Mg150K_GAAA = data_GAAA_5Mg150K_series.copy()
curated_data_5Mg150K_GAAA[curated_data_5Mg150K_GAAA>dG_threshold] = dG_threshold
curated_data_5Mg150K_GAAA = curated_data_5Mg150K_GAAA .dropna()
print('The average for all data taken (within the threshold) at 5 mM MG + 150K for GAAA is : ' + str(curated_data_5Mg150K_GAAA.mean()))
print('The std for all data taken (within the threshold) at 5 mM MG  + 150K for GAAA is : ' + str(curated_data_5Mg150K_GAAA.std()))
print('number of measurements :' + str(curated_data_5Mg150K_GAAA.size))
#%% What are the averages for each condition across the entire thing 
#   WHEN NOT TAKING INTO ACCOUN THE LIMITS.
curated_data_30Mg_GAAA = data_GAAA_30Mg_series.copy()
curated_data_30Mg_GAAA[curated_data_30Mg_GAAA>dG_threshold] = np.nan
curated_data_30Mg_GAAA = curated_data_30Mg_GAAA .dropna()
print('The average for all data taken (within the threshold) at 30 mM MG for GAAA is : ' + str(curated_data_30Mg_GAAA.mean()))
print('The std for all data taken (within the threshold) at 30 mM MG for GAAA is : ' + str(curated_data_30Mg_GAAA.std()))
print('number of measurements :' + str(curated_data_30Mg_GAAA.size))

curated_data_30Mg_GUAA = data_GUAA_30Mg_series.copy()
curated_data_30Mg_GUAA[curated_data_30Mg_GUAA>dG_threshold] = np.nan
curated_data_30Mg_GUAA = curated_data_30Mg_GUAA.dropna()
curated_data_30Mg_GUAA[curated_data_30Mg_GUAA>dG_threshold] = dG_threshold
print('The average for all data taken (within the threshold) at 30 mM MG for GUAA is : ' + str(curated_data_30Mg_GUAA.mean()))
print('The std for all data taken (within the threshold) at 30 mM MG for GUAA is : ' + str(curated_data_30Mg_GUAA.std()))
print('number of measurements :' + str(curated_data_30Mg_GUAA.size))


curated_data_5Mg_GAAA = data_GAAA_5Mg_series.copy()
curated_data_5Mg_GAAA[curated_data_5Mg_GAAA>dG_threshold] = np.nan
curated_data_5Mg_GAAA = curated_data_5Mg_GAAA .dropna()
print('The average for all data taken (within the threshold) at 5 mM MG for GAAA is : ' + str(curated_data_5Mg_GAAA.mean()))
print('The std for all data taken (within the threshold) at 5 mM MG for GAAA is : ' + str(curated_data_5Mg_GAAA.std()))
print('number of measurements :' + str(curated_data_5Mg_GAAA.size))

curated_data_5Mg150K_GAAA = data_GAAA_5Mg150K_series.copy()
curated_data_5Mg150K_GAAA[curated_data_5Mg150K_GAAA>dG_threshold] = np.nan
curated_data_5Mg150K_GAAA = curated_data_5Mg150K_GAAA .dropna()
print('The average for all data taken (within the threshold) at 5 mM MG + 150K for GAAA is : ' + str(curated_data_5Mg150K_GAAA.mean()))
print('The std for all data taken (within the threshold) at 5 mM MG  + 150K for GAAA is : ' + str(curated_data_5Mg150K_GAAA.std()))
print('number of measurements :' + str(curated_data_5Mg150K_GAAA.size))
#%%
# Scatter Plot of 30 mM vs. 5 mM for all data analyzed above in prep_data
low_lim = -14
high_lim = -6
dG_threshold = -7.1
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
plt.figure()
x_data = data_GAAA_30Mg_series
y_data = data_GAAA_5Mg_series
plt.scatter(x_data,y_data)
#calculate ddG_avg for values above threshold
x_data_thr = x_data[x_data<dG_threshold]
y_data_thr = y_data[y_data<dG_threshold]
ddG = y_data_thr - x_data_thr
ddG_avg = ddG.mean()
plt.plot(x,y_thres,':k',linewidth = 0.5)
plt.plot(y_thres,x,':k',linewidth = 0.5)
plt.plot(x,x,':k')
plt.plot(x,[element+ddG_avg for element in x],'--r')
plt.xlim(low_lim,high_lim)
plt.ylim(low_lim,high_lim)
plt.axes().set_aspect('equal')
#%%
# Scatter Plot of 30 mM vs. 5 mM + potassium
low_lim = -14
high_lim = -6
dG_threshold = -7.1
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
plt.figure()
x_data = data_GAAA_30Mg_series
y_data = data_GAAA_5Mg150K_series
plt.scatter(x_data,y_data)
#calculate ddG_avg for values above threshold
x_data_thr = x_data[x_data<dG_threshold]
y_data_thr = y_data[y_data<dG_threshold]
ddG = y_data_thr - x_data_thr
ddG_avg = ddG.mean()
plt.plot(x,y_thres,':k',linewidth = 0.5)
plt.plot(y_thres,x,':k',linewidth = 0.5)
plt.plot(x,x,':k')
plt.plot(x,[element+ddG_avg for element in x],'--r')
plt.xlim(low_lim,high_lim)
plt.ylim(low_lim,high_lim)
plt.axes().set_aspect('equal')
#%%
# Scatter Plot of 5mM mM vs. 5 mM + potassium
low_lim = -14
high_lim = -6
dG_threshold = -7.1
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
plt.figure()
x_data = data_GAAA_5Mg_series
y_data = data_GAAA_5Mg150K_series
plt.scatter(x_data,y_data)
#calculate ddG_avg for values above threshold
x_data_thr = x_data[x_data<dG_threshold]
y_data_thr = y_data[y_data<dG_threshold]
ddG = y_data_thr - x_data_thr
ddG_avg = ddG.mean()
plt.plot(x,y_thres,':k',linewidth = 0.5)
plt.plot(y_thres,x,':k',linewidth = 0.5)
plt.plot(x,x,':k')
plt.plot(x,[element+ddG_avg for element in x],'--r')
plt.xlim(low_lim,high_lim)
plt.ylim(low_lim,high_lim)
plt.axes().set_aspect('equal')
#%%
sublib0_analyzed = lib_analyzed[lib_analyzed.sublibrary == 'tertcontacts_0']
sublib3_analyzed = lib_analyzed[lib_analyzed.sublibrary == 'tertcontacts_3']
sublib2_analyzed = lib_analyzed[lib_analyzed.sublibrary == 'tertcontacts_2']
sublib4_analyzed = lib_analyzed[lib_analyzed.sublibrary == 'tertcontacts_4']
#%%
TL_specificity_df = pd.DataFrame(index=high_salt_data.index)
for scaffolds in scaffolds_five:
    specificity = high_salt_data['dG_30mM_Mg_GAAA_' + scaffolds] -\
    high_salt_data['dG_30mM_Mg_GUAA_' + scaffolds]
    TL_specificity_df['specificity_' + scaffolds] = specificity
TL_specificity_df['space'] = np.nan
TL_specificity_df['average'] = TL_specificity_df.mean(axis=1)

TL_specificity_SD = TL_specificity_df.std(axis=1)

cg_TL_specificity= sns.clustermap(TL_specificity_df,row_linkage=z_pca, col_cluster=False,
                        vmin=-4.5,vmax=4.5,cmap='coolwarm')
cg_TL_specificity.savefig('/Volumes/NO NAME/Clustermaps/clustermap_specificity.svg')

cg_TL_specificity_SD= sns.clustermap(TL_specificity_SD,row_linkage=z_pca, col_cluster=False,
                        vmin=0,vmax=1,cmap='Blues')
#%%
plt.figure()
markers = ['s','o','^','v','*']
for counter,scaffolds in enumerate(scaffolds_five):
    plt.scatter(data_50_scaffolds['dG_30mM_Mg_GAAA_' + scaffolds],
                data_50_scaffolds['dG_30mM_Mg_GUAA_' + scaffolds],s=120,edgecolors='k',marker='o')
plt.plot(x,x,':k')
plt.xlim(low_lim,high_lim)
plt.ylim(low_lim,high_lim)
plt.axes().set_aspect('equal')
#cg_TL_specificity.savefig('/Volumes/NO NAME/Clustermaps/specificty_heatmap.svg')
#cg_TL_specificity.savefig('/Volumes/NO NAME/Clustermaps/specificty_heatmap.svg')
#%% INFORMATION FOR CREATING TABLES DESCRIBING WHERE TL/TLRs CAME FROM 
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
frequency_sublib0 = pd.read_csv(data_path + 'sublib0_frequencies_08_03_2018.csv',header=None)
columns = ['three_prime','five_prime','name','found','table','confirmed']
frequency_sublib0.columns = columns
#frequency_sublib0['three_prime'] = frequency_sublib0['three_prime'].map(lambda x: x.lstrip('\'').rstrip(''))
frequency_sublib0['three_prime'] = frequency_sublib0['three_prime'].map(lambda x: x.strip('\''))
frequency_sublib0['five_prime'] = frequency_sublib0['five_prime'].map(lambda x: x.strip('\''))
frequency_sublib0['name'] = frequency_sublib0['name'].map(lambda x: x.strip('\''))
frequency_sublib0['new_name'] = frequency_sublib0['name'] + '_' +  frequency_sublib0['three_prime'] + '_' + frequency_sublib0['five_prime']
frequency_sublib0 = frequency_sublib0.set_index('new_name')
frequency_sublib0 ['r_seq'] = frequency_sublib0['three_prime'] + '_' + frequency_sublib0['five_prime']
frequency_sublib0 = frequency_sublib0.drop_duplicates(subset='r_seq')
#%%Function for comparing two strings of the same length
def compare(a, b):
    counter = 0
    for x, y in zip(a, b):
        if x != y:
            counter += 1
    return counter
#%%
#What are the receptors that made it into the heatmaps and clustermaps??
receptors_analyzed = list(prep_data_norm_with_nan.index)

#%%
#How many 11ntRs extracted from natural sequence databases??
#How many mutations relative to canonical sequence
canonical_11ntR = 'UAUGG_CCUAAG'
mutations = []
count_11ntR_found = 0
for sequences in frequency_sublib0.index:
    if '11ntR' in sequences:
        if sequences in receptors_analyzed:
            if frequency_sublib0.loc[sequences]['confirmed'] > 0:
                print(sequences)
                actual_sequence = receptor_info.loc[sequences]['r_seq']
                print(actual_sequence)
                count_11ntR_found += 1
                no_mutations = compare(actual_sequence,canonical_11ntR)
                mutations = mutations + [no_mutations]
print('there are ' + str(count_11ntR_found) + ' 11ntRs that were analyzed as part of\
      sublib0 and that are found in nature')
dummy_series = pd.Series(mutations)
#%%
count_IC3_found = 0
for sequences in frequency_sublib0.index:
    if 'IC3' in sequences:
        if sequences in receptors_analyzed:
            if frequency_sublib0.loc[sequences]['confirmed'] > 0:
                print(sequences)
                count_IC3_found += 1
print('there are ' + str(count_IC3_found) + ' IC3 that were analyzed as part of\
      sublib0 and that are found in nature')

#%%
count_C7_found = 0
for sequences in frequency_sublib0.index:
    if ('C7' in sequences) or ('B7' in sequences) or ('GAAC' in sequences):
        if sequences in receptors_analyzed:
            if frequency_sublib0.loc[sequences]['confirmed'] > 0:
                print(sequences)
                count_C7_found += 1
print('there are ' + str(count_C7_found) + ' IC3 that were analyzed as part of\
      sublib0 and that are found in nature')

#%%
count_C7_found = 0
for sequences in frequency_sublib0.index:
    if ('C7' in sequences) or ('B7' in sequences) or ('GAAC' in sequences):
        if sequences in receptors_analyzed:
            print(sequences)
            count_C7_found += 1
print('there are ' + str(count_C7_found) + ' in vitro that were analyzed as part of sublib0')

#%%
list_receptors_analyzed_sublib0 = []
for sequences in frequency_sublib0.index:
    if sequences in receptors_analyzed:
        list_receptors_analyzed_sublib0 = list_receptors_analyzed_sublib0 + [sequences]
        
dummy_receptors_df = receptor_types_df.copy()
dummy_receptors_df = dummy_receptors_df.reindex(list_receptors_analyzed_sublib0)

#%%Count number of 11ntR mutants in library analyzed
canonical_11ntR = 'UAUGG_CCUAAG'
mutations = []
#count number of single,double, and higher order mutants relative to canonical 11ntR
receptors_11ntR = receptor_info[receptor_info.type == '11ntR']
for each_sequence in receptors_11ntR.r_seq:
    no_mutations = compare(each_sequence,canonical_11ntR)
    mutations = mutations + [no_mutations]
receptors_11ntR['no_mutations'] = mutations
#%%Count number of IC3 mutants in library analyzed
canonical_IC3 = 'GAGGG_CCCUAAC'
mutations = []
#count number of single,double, and higher order mutants relative to canonical 11ntR
receptors_IC3 = receptor_info[receptor_info.type == 'IC3']
for each_sequence in receptors_IC3.r_seq:
    no_mutations = compare(each_sequence,canonical_IC3)
    mutations = mutations + [no_mutations]
receptors_IC3['no_mutations'] = mutations

#%%Count number of VC2 mutants in library analyzed
canonical_VC2 = 'GUAGG_CCUAAC'
mutations = []
#count number of single,double, and higher order mutants relative to canonical 11ntR
receptors_VC2 = receptor_info[receptor_info.type == 'VC2']
for each_sequence in receptors_VC2.r_seq:
    no_mutations = compare(each_sequence,canonical_VC2)
    mutations = mutations + [no_mutations]
receptors_VC2['no_mutations'] = mutations
  
#%%
mask = lib_analyzed.dG_Mut2_GAAA.isnull() & lib_analyzed.dG_Mut2_GAAA_5mM_2.isnull() & lib_analyzed.dG_Mut2_GAAA_5mM_150mMK_1.isnull() & lib_analyzed.dG_Mut2_GUAA_1.isnull()
mask2 = lib_analyzed.dG_Mut2_GAAA.isnull()
temp  = lib_analyzed[mask2]


#%%
scaff_35311 = entire_lib_selected[entire_lib_selected.old_idx == '35311_A']
scaff_14073 = entire_lib_selected[entire_lib_selected.old_idx == '14073']
scaff_35600 = entire_lib_selected[entire_lib_selected.old_idx == '35600']
scaff_14007 = entire_lib_selected[entire_lib_selected.old_idx == '14007']
scaff_13854 = entire_lib_selected[entire_lib_selected.old_idx == '13854']

scaff_13854 = scaff_13854.set_index('r_seq')
scaff_14073 = scaff_14073.set_index('r_seq')
scaff_35600 = scaff_35600.set_index('r_seq')
scaff_14007 = scaff_14007.set_index('r_seq')
scaff_35311 = scaff_35311.set_index('r_seq')

scaff_14073 = scaff_14073.reindex(scaff_13854.index)
plt.figure()
plt.scatter(scaff_14073.dG_Mut2_GAAA,scaff_13854.dG_Mut2_GAAA)
plt.plot(scaff_14073.dG_Mut2_GAAA,scaff_14073.dG_Mut2_GAAA,'--k')

plt.figure()
plt.scatter(scaff_14073.dG_Mut2_GAAA_5mM_150mMK_1,scaff_13854.dG_Mut2_GAAA_5mM_150mMK_1)
plt.plot(scaff_14073.dG_Mut2_GAAA,scaff_14073.dG_Mut2_GAAA,'--k')


#%%
sublib0_ver2 = sublib0.set_index('r_seq')
for sequences in sublib0_ver2.index[1:10]:
    plt.figure()
    plt.scatter(sublib0_ver2.loc[sequences].dG_Mut2_GAAA_5mM_2,sublib0_ver2.loc[sequences].dG_Mut2_GAAA_5mM_150mMK_1)
    plt.xlim(-14,-7)
    plt.ylim(-14,-7)
    plt.plot([-14,-7],[-14,-7],'--r')
