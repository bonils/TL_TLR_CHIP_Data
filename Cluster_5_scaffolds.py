
# coding: utf-8

# Description of project:
# 
# Tetraloop/tetraloop-receptors (TL/TLR) are modular components of RNA structure consisting of a tetraloop (TL; the nucleotide sequence of the TL is semi-conserved between different TL/TLRs) making interactions with a receptor helix. TL/TLRs stabilize the 3D structure of diverse RNAs including small ribozymes, riboswitches, RNase P, and the ribosome by bringing helices together. Different types of TL/TLR interactions are observed in sequence and structural databases, and they have been categorized based on their structural complexity (i.e. the number of residues involved).
# 
# We have used a high-throuput assay to measure the thermodynamic stability of > 1000 TL/TLR sequence variants. Each TL/TLR sequence variant was embedded into multiple structural scaffolds and characterized under different solution conditions. For this project, I worked with a subset of the data: five structural scaffolds, three solution conditions, and one alternative TL. Though in this project I am analyzed the stability of each TL/TLR accross 20 'perturbations'.
# 
# The raw data are divided into TL/TLR belonging to three structural categories: (1) 11ntR-like, in vitro selected, and IC3/VC2-like.  
# 
# The objective of this project is to categorize the TL/TLRs in terms of their thermodynamic stability and their sensitivity to the structural and solution perturbations. Our working model is that thermodynamically and conformationally similar TL/TLRs will have show a similar 'profile' of stability accross the perturbations. 
# 
# The steps that I followed are:
# 
# (1) Extract subset of data for each TL/TLR.
# (2) Eliminate TL/TLRs (rows) that had too few data or that had too many unstable meausurements (threshold is -7.1 kcal/mol, stability values above the threshold are not reliable).
# (3) For the rest of missing data, use information fropm nearest neighbors to fill in missing data.
# (4) Replace values that are above threshold with -7.1 kcal/mol (i.e. limits).
# (5) Normalize data.
# (7) Principal component analysis to reduce the number of dimensions for downstream clustering. 
# (8) Hierarchical clustering of the PCs. 
# (9) Determine the number of clusters from dendrogram (for now just by visually inspecting the dendrogram and choosing a distance threshold).
# (10) Analyze each cluster individually and plot summaries describing behavior. 

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
#%%
#General Variables
dG_threshold = -7.1 #kcal/mol; dG values above this are not reliable
dG_replace = -7.1 # for replacing values above threshold. 
nan_threshold = 0.50 #amount of missing data tolerated.
#%%
#Define Functions
def get_dG_for_scaffold(data,scaffold,conditions,row_index,column_labels,flanking ='normal'):
    #Function for getting dG info for a particular scaffold.
    #Arguments:
    #data: dataframe containing TL/TLR information.
    #scaffold: the identifier of the chip scaffolds to include in analysis; scaffold= data.old_idx (integer)
    #conditions: dG conditions (columns) to pull from data dataframe(list of strings)
    #row_index: column to use for indexing
    #column_labels:labels for the columns of the output dataframe (list of strings)
    #flanking: the base pair flaking the TLR: default is 'normal' (string)
    
    data_scaffold = data.loc[(data.old_idx == scaffold)
                                                & (data.b_name == flanking)]
    data_scaffold = data_scaffold.set_index(row_index)
    data_scaffold_dG = data_scaffold[conditions]
    data_scaffold_dG.columns = column_labels
    print('Size of dG matrix for scaffold ', str(scaffold),data_scaffold_dG.shape)
    return data_scaffold_dG

def doPCA(data):
    #Function for performing PCA on matrix
    PCA = skd.PCA(n_components=None, whiten=False)
    transformed_data = PCA.fit_transform(data)
    principal_comp = np.dot(np.linalg.inv(np.dot(transformed_data.T, transformed_data)),
                                 np.dot(transformed_data.T, data))
    
    # put in terms of pandas dataframe
    transformed_data = pd.DataFrame(transformed_data, index=data.index,
                                    columns=['pc_%d'%i for i in np.arange(transformed_data.shape[1])])
    principal_comp = pd.DataFrame(principal_comp, columns=data.columns,
                                  index=['pc_%d'%i for i in np.arange(transformed_data.shape[1])])
    
    return (PCA, transformed_data, principal_comp)


#Function for interpolating data 

def interpolate_mat_knn(data, min_num_to_average=1, max_num_to_average=20,
                        threshold=0.2, enforce_threshold=False, metric='cityblock'):
    #Function for filling nans with value of nearest neighbor column 
    num_cols = data.shape[1]
    average_distance = {}
    num_cols_not_nan = {}
    num_neighbors = {}
    data_interp = []

    for name, group in data.isnull().groupby([col for col in data]):
        # find the subgroup with nans in these columns
        index_sub = group.index.tolist()
        newdata = data.loc[index_sub].copy()
        cols = ~pd.Series(list(name), index=data.columns)
        
        # if there are no nans, continue wihtout interpolating
        num_nan = np.sum(~cols)
        if num_nan==0:
            data_interp.append(newdata)
            for idx_current in newdata.index.tolist():
                average_distance[idx_current] = np.nan
                num_cols_not_nan[idx_current] = cols.sum()
                num_neighbors[idx_current] = 0
            continue
        
        if num_nan==num_cols:
            # no data to compare with
            continue

        # find the training set to compare to, and nearest neighbors
        index_all = data.loc[:, cols].dropna().index.tolist()
        index_all = data.loc[index_all, ~cols].dropna(how='all').index.tolist()
        nbrs = NearestNeighbors(n_neighbors=max_num_to_average, algorithm='ball_tree',
                                metric=metric).fit(data.loc[index_all, cols])
        distances, indices = nbrs.kneighbors(newdata.loc[:, cols])
        
        # for every value in newdata, find nearest neighbors and interpolate
        for i, (index_vec_i, distance_vec_i) in enumerate(zip(indices, distances)):
            idx_current = index_sub[i]
            index_vec = [index_all[j] for j in index_vec_i]
            distance_vec_sub = pd.Series({idx:d/cols.sum() for idx,
                                          d in zip(index_vec, distance_vec_i) if idx!=idx_current}).sort_values()
            index_to_average = distance_vec_sub.loc[distance_vec_sub < threshold].index.tolist()
            ## pair down indices that have no values in the OTHER cols of data
            #if np.any(data.loc[index_to_average, ~cols].mean().isnull()):
            #    distance_vec_sub = distance_vec_sub.loc[[idx for idx in distance_vec_sub.index.tolist() if idx not in index_to_average]]
            #    index_to_average = []

            # if there are fewer indices than lower threshold, dcide whether to increase threshold or enforce
            if len(index_to_average)<min_num_to_average:
                if enforce_threshold:
                    index_to_average = []
                else:
                    # no entries within threshold: find 5 closest
                    index_to_average = distance_vec_sub.index.tolist()[:min_num_to_average]            
            average_distance[idx_current] = distance_vec_sub.loc[index_to_average].mean()
            num_cols_not_nan[idx_current] = cols.sum()
            num_neighbors[idx_current] = len(index_to_average)
            newdata.loc[idx_current, ~cols] = data.loc[index_to_average, ~cols].mean()
        data_interp.append(newdata)
    
    data_interp = pd.concat(data_interp)
    num_cols_not_nan = pd.Series(num_cols_not_nan)
    num_neighbors = pd.Series(num_neighbors)
    average_distance = pd.Series(average_distance)
    interp_info = pd.concat([average_distance.rename('average_distance'), 
                             num_neighbors.rename('num_neighbors'), num_cols_not_nan.rename('number_of_columns')], axis=1)
    return data_interp, interp_info


def prep_data_for_clustering_ver2 (data,dG_thr, dG_rep, max_nan):
    #(1) delete rows that are either all below threshold or Nan
    A = data.copy()
    crappy_mask = (A.isna()) | (A> dG_thr)
    row_mask = crappy_mask.all(axis=1)
    print('Size of dataframe before eliminating rows that are entirely NaN or above dG_threshold:',
          A.shape)
    A = A.loc[~row_mask]
    print('Size of dataframe after eliminating rows that are entirely NaN or above dG_threshold :',
          A.shape)
    #(2) Eliminate rows that have too many NaN according to max_nan
    nan_per_row = A.isnull().sum(axis=1)
    num_columns = len(A.columns)
    percent_nan_per_row = nan_per_row/num_columns
    row_mask = percent_nan_per_row > max_nan
    A = A.loc[~row_mask]
    print('Size after eliminating rows that have more NaNs than threshold:', A.shape)
    #(3) Fill missing values, interpolating using nearest neighbors
    original_NaNs = A.isnull() # Keep track of NaNs that were interpolated
    tot_nan_before = A.isnull().sum(axis=1).sum(axis=0)
    print('Initial number of missing data before interpolation: ', tot_nan_before)
    if tot_nan_before > 0:
        tot_nan = tot_nan_before
        B = A.copy()
        counter = 0
        while tot_nan > 0:
            counter += 1
            print('number of interpolations :',counter)
            B,info = interpolate_mat_knn(B)
            tot_nan = B.isnull().sum(axis=1).sum(axis=0)
            print('there are ', tot_nan, ' missing data left')
    else:
        B = A.copy()
        print('No need for interpolation, no missing data')
    tot_nan_after = B.isnull().sum(axis=1).sum(axis=0)
    print('Number of missing data after interpolation: ', tot_nan_after)
    print('--' * 50)
    print('replacing dG values > ', dG_thr, ' kcal/mol by ', dG_rep, ' kcal/mol.')
    #(4) Replace values above threshold by limit
    B[B > dG_thr] = dG_rep 
    
    C = B.copy()
    original_NaNs = original_NaNs.reindex(C.index)
    return B,original_NaNs


#%%
#import data from csv files 
#Data has been separated into 11ntR, IC3, and in vitro

data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'

#11ntR
all_11ntRs_unique_df = pd.read_csv(data_path + 'all_11ntRs_unique.csv')
unique_receptors_11ntR = sorted(list(set(list(all_11ntRs_unique_df.r_seq))))#used for indexing 
print(unique_receptors_11ntR[0])
print('# of unique 11ntR receptors: '+ str(len(unique_receptors_11ntR)))

#IC3
all_IC3_unique_df = pd.read_csv(data_path + 'all_IC3_unique.csv')
unique_receptors_IC3 = sorted(list(set(list(all_IC3_unique_df.r_name_seq))))#used for indexing
print('# of unique IC3 receptors: '+ str(len(unique_receptors_IC3)))

#in vitro
all_in_vitro_unique_df = pd.read_csv(data_path + 'all_in_vitro_unique.csv')
all_in_vitro_unique_df['new_name'] = all_in_vitro_unique_df['r_name'] + '_' + all_in_vitro_unique_df['r_seq']
unique_receptors_in_vitro = sorted(list(set(list(all_in_vitro_unique_df.new_name)))) #used for indexing
print('# of unique in vitro receptors: '+ str(len(unique_receptors_in_vitro)))

#%%
#---------------------------PREPARE DATA FOR 11nTR TLRS--------------------------------------------

# (1) Select data for five different structural scaffolds for which we have data for all variants.
#     These are old_idx 35600;35311;14007;14073;13854
# (2) Index with respect to receptor sequence (r_seq)


#Common to the five scaffolds for 11ntR
data = all_11ntRs_unique_df
conditions = ['dG_Mut2_GAAA','dG_Mut2_GAAA_5mM_2','dG_Mut2_GAAA_5mM_150mMK_1','dG_Mut2_GUAA_1']
row_index= ('r_seq')
flanking = 'normal'

#Call get_dG_for_scaffold for each of the five scaffolds
scaffold = 35600
column_labels = ['dG_30Mg_GAAA_35600','dG_5Mg_GAAA_35600','dG_5Mg150K_GAAA_35600','dG_30Mg_GUAA_35600'] 
all_11ntRs_35600_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking)
all_11ntRs_35600_dG = all_11ntRs_35600_dG.reindex(unique_receptors_11ntR)

scaffold = 35311
column_labels = ['dG_30Mg_GAAA_35311','dG_5Mg_GAAA_35311','dG_5Mg150K_GAAA_35311','dG_30Mg_GUAA_35311'] 
all_11ntRs_35311_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking) 
all_11ntRs_35311_dG = all_11ntRs_35311_dG.reindex(unique_receptors_11ntR)

scaffold = 14007
column_labels = ['dG_30Mg_GAAA_14007','dG_5Mg_GAAA_14007','dG_5Mg150K_GAAA_14007','dG_30Mg_GUAA_14007'] 
all_11ntRs_14007_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking) 
all_11ntRs_14007_dG = all_11ntRs_14007_dG.reindex(unique_receptors_11ntR)

scaffold = 14073
column_labels = ['dG_30Mg_GAAA_14073','dG_5Mg_GAAA_14073','dG_5Mg150K_GAAA_14073','dG_30Mg_GUAA_14073'] 
all_11ntRs_14073_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking) 
all_11ntRs_14073_dG = all_11ntRs_14073_dG.reindex(unique_receptors_11ntR)

scaffold = 13854
column_labels = ['dG_30Mg_GAAA_13854','dG_5Mg_GAAA_13854','dG_5Mg150K_GAAA_13854','dG_30Mg_GUAA_13854'] 
all_11ntRs_13854_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking) 
all_11ntRs_13854_dG = all_11ntRs_13854_dG.reindex(unique_receptors_11ntR)

all_11ntRs_five_scaffolds = pd.concat([all_11ntRs_35600_dG,all_11ntRs_35311_dG,
                                       all_11ntRs_14007_dG,all_11ntRs_14073_dG,all_11ntRs_13854_dG],axis =1)
all_11ntRs_five_scaffolds.shape


#---------------------------PREPARE DATA FOR IC3 TLRS--------------------------------------------

# (1) Select data for five different structural scaffolds for which we have data for all variants.
#     These are old_idx 35600;35311;14007;14073;13854
# (2) Index with respect to receptor sequence (r_name_seq)


#Common to the five scaffolds for IC3
data = all_IC3_unique_df
conditions = ['dG_Mut2_GAAA','dG_Mut2_GAAA_5mM_2','dG_Mut2_GAAA_5mM_150mMK_1','dG_Mut2_GUAA_1']
row_index= ('r_name_seq')
flanking = 'normal'

#Call get_dG_for_scaffold for each of the five scaffolds
scaffold = 35600
column_labels = ['dG_30Mg_GAAA_35600','dG_5Mg_GAAA_35600','dG_5Mg150K_GAAA_35600','dG_30Mg_GUAA_35600'] 
all_IC3_35600_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking) 
all_IC3_35600_dG = all_IC3_35600_dG.reindex(unique_receptors_IC3)

scaffold = 35311
column_labels = ['dG_30Mg_GAAA_35311','dG_5Mg_GAAA_35311','dG_5Mg150K_GAAA_35311','dG_30Mg_GUAA_35311'] 
all_IC3_35311_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking) 
all_IC3_35311_dG = all_IC3_35311_dG.reindex(unique_receptors_IC3)

scaffold = 14007
column_labels = ['dG_30Mg_GAAA_14007','dG_5Mg_GAAA_14007','dG_5Mg150K_GAAA_14007','dG_30Mg_GUAA_14007'] 
all_IC3_14007_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking) 
all_IC3_14007_dG = all_IC3_14007_dG.reindex(unique_receptors_IC3)

scaffold = 14073
column_labels = ['dG_30Mg_GAAA_14073','dG_5Mg_GAAA_14073','dG_5Mg150K_GAAA_14073','dG_30Mg_GUAA_14073'] 
all_IC3_14073_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking)
all_IC3_14073_dG = all_IC3_14073_dG.reindex(unique_receptors_IC3)

scaffold = 13854
column_labels = ['dG_30Mg_GAAA_13854','dG_5Mg_GAAA_13854','dG_5Mg150K_GAAA_13854','dG_30Mg_GUAA_13854'] 
all_IC3_13854_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking)
all_IC3_13854_dG = all_IC3_13854_dG.reindex(unique_receptors_IC3)

#Concatanete the five scaffolds, sorting with respect to the list of unique sequences created above 
all_IC3_five_scaffolds = pd.concat([all_IC3_35600_dG,all_IC3_35311_dG,
                                       all_IC3_14007_dG,all_IC3_14073_dG,all_IC3_13854_dG],axis =1)
print(all_IC3_five_scaffolds.shape)


#---------------------------PREPARE DATA FOR in vitro TLRS--------------------------------------------

# (1) Select data for five different structural scaffolds for which we have data for all variants.
#     These are old_idx 35600;35311;14007;14073;13854
# (2) Index with respect to receptor sequence (r_name_seq)


#Common to the five scaffolds for in vitro
data = all_in_vitro_unique_df
conditions = ['dG_Mut2_GAAA','dG_Mut2_GAAA_5mM_2','dG_Mut2_GAAA_5mM_150mMK_1','dG_Mut2_GUAA_1']
row_index= ('new_name')
flanking = 'normal'

#Call get_dG_for_scaffold for each of the five scaffolds
scaffold = 35600
column_labels = ['dG_30Mg_GAAA_35600','dG_5Mg_GAAA_35600','dG_5Mg150K_GAAA_35600','dG_30Mg_GUAA_35600'] 
all_in_vitro_35600_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking) 
all_in_vitro_35600_dG = all_in_vitro_35600_dG.reindex(unique_receptors_in_vitro)

scaffold = 35311
column_labels = ['dG_30Mg_GAAA_35311','dG_5Mg_GAAA_35311','dG_5Mg150K_GAAA_35311','dG_30Mg_GUAA_35311'] 
all_in_vitro_35311_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking)
all_in_vitro_35311_dG = all_in_vitro_35311_dG.reindex(unique_receptors_in_vitro)

scaffold = 14007
column_labels = ['dG_30Mg_GAAA_14007','dG_5Mg_GAAA_14007','dG_5Mg150K_GAAA_14007','dG_30Mg_GUAA_14007'] 
all_in_vitro_14007_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking)
all_in_vitro_14007_dG = all_in_vitro_14007_dG.reindex(unique_receptors_in_vitro)

scaffold = 14073
column_labels = ['dG_30Mg_GAAA_14073','dG_5Mg_GAAA_14073','dG_5Mg150K_GAAA_14073','dG_30Mg_GUAA_14073'] 
all_in_vitro_14073_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking) 
all_in_vitro_14073_dG = all_in_vitro_14073_dG.reindex(unique_receptors_in_vitro)

scaffold = 13854
column_labels = ['dG_30Mg_GAAA_13854','dG_5Mg_GAAA_13854','dG_5Mg150K_GAAA_13854','dG_30Mg_GUAA_13854'] 
all_in_vitro_13854_dG = get_dG_for_scaffold(data,scaffold,conditions, row_index, column_labels,flanking) 
all_in_vitro_13854_dG = all_in_vitro_13854_dG.reindex(unique_receptors_in_vitro)

all_in_vitro_five_scaffolds = pd.concat([all_in_vitro_35600_dG,all_in_vitro_35311_dG,
                                       all_in_vitro_14007_dG,all_in_vitro_14073_dG,all_in_vitro_13854_dG],axis =1)
                                       
print(all_in_vitro_five_scaffolds.shape)


#%%
#Prepare data for clustering and plotting
#(1) Combine data from all Tl/TLRs; REARRANGE COLUMNS
combined_data_rearr = pd.concat([all_11ntRs_five_scaffolds,all_IC3_five_scaffolds,all_in_vitro_five_scaffolds])
columns_reordered = ['dG_30Mg_GAAA_35600', 'dG_30Mg_GAAA_35311', 'dG_30Mg_GAAA_14007','dG_30Mg_GAAA_14073','dG_30Mg_GAAA_13854',
                     'dG_5Mg_GAAA_35600','dG_5Mg_GAAA_35311','dG_5Mg_GAAA_14007', 'dG_5Mg_GAAA_14073', 'dG_5Mg_GAAA_13854',
                     'dG_5Mg150K_GAAA_35600', 'dG_5Mg150K_GAAA_35311', 'dG_5Mg150K_GAAA_14007', 'dG_5Mg150K_GAAA_14073', 'dG_5Mg150K_GAAA_13854',
                     'dG_30Mg_GUAA_35600', 'dG_30Mg_GUAA_35311', 'dG_30Mg_GUAA_14007', 'dG_30Mg_GUAA_14073','dG_30Mg_GUAA_13854']
combined_data_rearr = combined_data_rearr[columns_reordered]

#(2) Eliminate rows with too much missing data, fill remaining missing data with nearest neighbors info,
# replace values above threshold with limit
prep_data,original_nan = prep_data_for_clustering_ver2(combined_data_rearr,-7.1,-7.1,0.5)

#(3) Subtract the mean per row
norm_data = prep_data.sub(prep_data.mean(axis = 1),axis = 0)
norm_data_nan = norm_data.copy()
norm_data_nan[original_nan] = np.NAN
#%%
#PCA Analyis
pca,transformed,loadings = doPCA(norm_data)
#plot explained variance by PC
pd.Series(pca.explained_variance_ratio_).plot(kind='bar')
plt.ylabel('fraction of variance \nexplained by each PC', fontsize=14)
plt.tight_layout()
num_PCA = 4
print('Fraction explained by the first ',str(num_PCA), 'PCAs :',sum(pca.explained_variance_ratio_[:num_PCA]))
plt.show()
# In[15]:


#Hierarchical clustering of the first 4 PCs
list_PCAs = list(transformed.columns[:num_PCA])
z_pca = sch.linkage(transformed.loc[:,list_PCAs],method='ward') 
cg_pca = sns.clustermap(norm_data_nan,row_linkage=z_pca, col_cluster=False, cmap='coolwarm', vmin=-4,vmax=4)
X = transformed.loc[:,list_PCAs]
c, coph_dists = cophenet(z_pca, pdist(X))
print('cophenetic distance: ',c)
show()
cg_pca.savefig('clustermap_no1.svg')


# In[16]:


#Plot dendrogram and get clusters based on threshold distance
sch.dendrogram(z_pca,color_threshold=15)
max_d = 15
clusters = fcluster(z_pca, max_d, criterion='distance')
number_clusters = max(clusters)
print('number of clusters based on distance of ',str(max_d), ':', str(number_clusters))
plt.show()


# In[18]:


# Append cluster number to dataframe with data
clustered_data = norm_data.copy()
clustered_data_nan = norm_data_nan.copy()
cluster_series = pd.Series(clusters,index=clustered_data.index)
clustered_data['cluster'] = cluster_series
clustered_data_nan['cluster'] = cluster_series


# In[26]:


#Replot clustergram with rows colored as above
row_color = pd.Series(clusters,index=clustered_data.index)
cluster_colors = ['g','r','c','m','y','k','b','orange','g','r','c','m','y','k','b']
num_clusters = len(cluster_series.unique())
num_clusters
for i in range(num_clusters):
    row_color[row_color == (i+1)] = cluster_colors[i]
cg_pca_col = sns.clustermap(norm_data_nan,row_linkage=z_pca, col_cluster=False, cmap='coolwarm', vmin=-4,vmax=4,row_colors=row_color)
cg_pca_col.savefig('clustermap.svg')                 


# In[14]:


#Append type and sequence of TLRs
receptors = clustered_data.index
types = []
sequence = []
for names in receptors:        
    if 'C7' in names or 'B7' in names:
        types.append(2)
        N = names.split('_')
        sequence.append(N[-2] + '_' + N[-1])

    elif 'IC3' in names or 'VC2' in names:
        types.append(3)
        N = names.split('_')
        sequence.append(N[-2] + '_' + N[-1])
    else:
        types.append(1)
        sequence.append(names)
types_series = pd.Series(types,index = clustered_data.index)
sequence_series = pd.Series(sequence,index = clustered_data.index)
clustered_data['receptor_type'] = types_series
clustered_data['r_seq'] = sequence_series
clustered_data_nan['receptor_type'] = types_series
clustered_data_nan['r_seq'] = sequence_series


# In[15]:


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


# In[16]:


#create dictionary for each cluster with original dG values before interpolation, and replacement of dG values
# dG values are in combined_data_rearr
cluster_names = ['cluster_' + str(i) for i in range(1,number_clusters+1)]
A = combined_data_rearr.copy()
all_clusters_dG = {}
for names in cluster_names:
    all_clusters_dG[names] = A.loc[all_clusters[names].index]


# In[53]:


#plot data summaries for each cluster
num_clusters = len(all_clusters)
#variables
dG_threshold = -7.1

#Colors to use in plotting
cluster_names = list(range(1,num_clusters + 1,1))
cluster_colors = ['g','r','c','m','y','b','g','r','c','m','y','b']
color_list = dict(zip(cluster_names,cluster_colors))

## Plot Summaries for each of the clusters for clustering analysis perfomed in Tl/TLR_analysis notebook
## Individual clusters were saved in a dictionary; one with the 'normalized' ddG values and another with dG values.
## The dataframes for each of the clusters are called by e.g. all_clusters['cluster_1]

for counter,clusters in enumerate(sorted(all_clusters)):

    print(clusters)
    
    fig = plt.figure()
    #For each of the clusters in ddG dictionary make copy
    next_cluster = all_clusters[clusters].copy()
    #Select columns with ddG values 
    next_cluster_values = next_cluster[next_cluster.columns[0:20]]
    #Rename columns with numerical 1 -20 for easier plotting
    next_cluster_values.columns = range(1,21,1)
    # Do the same as above for dG dictionary
    next_cluster_dG = all_clusters_dG[clusters].copy()
    next_cluster_dG.columns = range(1,21,1)
    
    #plot heatmap with ddG values for each cluster.
    ax1 = fig.add_subplot(3,2,1)
    sns.heatmap(next_cluster_values,vmax=4,vmin=-4,xticklabels=False, yticklabels=False,cmap='coolwarm',ax=ax1)
    ax1.set_title('$\Delta\Delta$G (kcal/mol)')
    #plot ddG profiles as lines
    ax2 = fig.add_subplot(3,2,2); 
    (s1,s2) = next_cluster.shape
    cluster_title = ('cluster ' + str(counter + 1) + '   n = ' + str(s1))
    next_cluster_values.T.plot(ax=ax2,legend=False, title= cluster_title,marker='o',xticks=list(range(5,21,5)),grid=True,
                        color=color_list[counter + 1],ylim=(-4,4))
    #plot heatmap with dG values 
    ax3 = fig.add_subplot(3,2,3); 
    sns.heatmap(next_cluster_dG,vmax=-7,vmin=-12,yticklabels=False,xticklabels=False,ax=ax3)
    ax3.set_title('$\Delta$G (kcal/mol)')
    
    #plot dG profiles as lines
    ax4 = fig.add_subplot(3,2,4); 
    next_cluster_dG.T.plot(legend=False, marker='o',xticks=list(range(5,21,5)),grid=True,
                    color=color_list[counter + 1],ylim=(-12,0),ax = ax4)
    limitX = [1,20]
    limitY = [-7.1,-7.1]
    plt.plot(limitX,limitY,'k--',linewidth=3.0)
    
    #Create mask indicating values that were below stability threshold or NaN and plot heatmap. 
    ax5 = fig.add_subplot(3,2,5);  
    next_mask = next_cluster_dG <dG_threshold
    sns.heatmap(next_mask.astype('int64'),yticklabels=False, xticklabels=False,cmap='gray',ax=ax5)
    
    #Create bar plot indicating the number of receptors of each type. 
    ax6 = fig.add_subplot(3,2,6)
    number_types = [sum(next_cluster.receptor_type  == 1),sum(next_cluster.receptor_type  == 2),sum(next_cluster.receptor_type  == 3)]
    receptor_types = ['11ntR-like', 'in vitro', 'IC3/VC2-like']
    plt.bar([1,2,3],number_types,align='center')
    plt.ylim([0,250])
    plt.xticks([1,2,3],receptor_types)
    fig_name = 'cluster ' + str(counter + 1) + 'summary.pdf'
    


# Repeat data analysis performed above but subtract WT 11ntR (reference) from each row instead of the mean of each row. 

# In[28]:


#Use one row (11ntR WT) as reference: substract it from all other rows
norm_data_Wt = prep_data.copy()
reference = norm_data_Wt.loc['UAUGG_CCUAAG']
norm_data_Wt = norm_data_Wt - reference
norm_data_Wt_nan = norm_data_Wt.copy()
norm_data_Wt_nan[original_nan] = np.NAN
print('Max value after subtracting is : ', norm_data_Wt.max().max())
print('Min value after subtracting is : ', norm_data_Wt.min().min())


# In[25]:


#PCA Analyis
pca,transformed,loadings = doPCA(norm_data_Wt)
#plot explained variance by PC
pd.Series(pca.explained_variance_ratio_).plot(kind='bar')
plt.ylabel('fraction of variance \nexplained by each PC', fontsize=14)
plt.tight_layout()
num_PCA = 4
print('Fraction explained by the first ',str(num_PCA), 'PCAs (data relative to 11ntR) :',sum(pca.explained_variance_ratio_[:num_PCA]))
show()


# In[29]:


#Hierarchical clustering of the first 4 PCs
list_PCAs = list(transformed.columns[:num_PCA])
z_pca = sch.linkage(transformed.loc[:,list_PCAs],method='ward') 
cg_pca = sns.clustermap(norm_data_Wt_nan,row_linkage=z_pca, col_cluster=False, cmap='coolwarm', vmin=-5,vmax=5)
X = transformed.loc[:,list_PCAs]
c, coph_dists = cophenet(z_pca, pdist(X))
print('cophenetic distance: ',c)
show()


# In[35]:


#Plot dendrogram and get clusters based on threshold distance
sch.dendrogram(z_pca,color_threshold=20)
max_d = 20
clusters = fcluster(z_pca, max_d, criterion='distance')
number_clusters = max(clusters)
print('number of clusters based on distance of ',str(max_d), ':', str(number_clusters))
plt.show()


# In[36]:


# Append cluster number to dataframe with data
clustered_data_Wt = norm_data_Wt.copy()
clustered_data_Wt_nan = norm_data_Wt_nan.copy()
cluster_series = pd.Series(clusters,index=clustered_data.index)
clustered_data_Wt['cluster'] = cluster_series
clustered_data_Wt_nan['cluster'] = cluster_series


# In[37]:


#Append type and sequence of TLRs
receptors = clustered_data_Wt.index
types = []
sequence = []
for names in receptors:        
    if 'C7' in names or 'B7' in names:
        types.append(2)
        N = names.split('_')
        sequence.append(N[-2] + '_' + N[-1])

    elif 'IC3' in names or 'VC2' in names:
        types.append(3)
        N = names.split('_')
        sequence.append(N[-2] + '_' + N[-1])
    else:
        types.append(1)
        sequence.append(names)
types_series = pd.Series(types,index = clustered_data_Wt.index)
sequence_series = pd.Series(sequence,index = clustered_data_Wt.index)
clustered_data_Wt['receptor_type'] = types_series
clustered_data_Wt['r_seq'] = sequence_series
clustered_data_Wt_nan['receptor_type'] = types_series
clustered_data_Wt_nan['r_seq'] = sequence_series


# In[38]:


#separate each cluster and save in dictionary
cluster_names = ['cluster_' + str(i) for i in range(1,number_clusters+1)]
all_clusters_Wt = {}
all_clusters_Wt_nan = {}
for counter, names in enumerate (cluster_names):
    all_clusters_Wt[names] = clustered_data_Wt.query('cluster==' + str(counter +1))
    all_clusters_Wt_nan[names] = clustered_data_Wt_nan.query('cluster==' + str(counter +1))
#how many members are in each cluster
for clusters in all_clusters_Wt:
    s1,s2 = all_clusters_Wt[clusters].shape
    print('There are ',str(s1),' tetraloop-receptors in ', clusters)


# In[42]:


#create dictionary for each cluster with original dG values before interpolation, and replacement of dG values
# dG values are in combined_data_rearr
cluster_names = ['cluster_' + str(i) for i in range(1,number_clusters+1)]
A = combined_data_rearr.copy()
all_clusters_dG_Wt = {}
for names in cluster_names:
    all_clusters_dG_Wt[names] = A.loc[all_clusters_Wt[names].index]


# In[ ]:


#plot data summaries for each cluster
num_clusters = len(all_clusters)
#variables
dG_threshold = -7.1

#Colors to use in plotting
cluster_names = list(range(1,num_clusters + 1,1))
cluster_colors = ['g','r','c','m','y','b','g','r','c','m','y','b']
color_list = dict(zip(cluster_names,cluster_colors))

## Plot Summaries for each of the clusters for clustering analysis perfomed in Tl/TLR_analysis notebook
## Individual clusters were saved in a dictionary; one with the 'normalized' ddG values and another with dG values.
## The dataframes for each of the clusters are called by e.g. all_clusters['cluster_1]

for counter,clusters in enumerate(sorted(all_clusters_Wt)):

    print(clusters)
    
    fig = plt.figure()
    #For each of the clusters in ddG dictionary make copy
    next_cluster = all_clusters_Wt[clusters].copy()
    #Select columns with ddG values 
    next_cluster_values = next_cluster[next_cluster.columns[0:20]]
    #Rename columns with numerical 1 -20 for easier plotting
    next_cluster_values.columns = range(1,21,1)
    # Do the same as above for dG dictionary
    next_cluster_dG = all_clusters_dG_Wt[clusters].copy()
    next_cluster_dG.columns = range(1,21,1)
    
    #plot heatmap with ddG values for each cluster.
    ax1 = fig.add_subplot(3,2,1)
    sns.heatmap(next_cluster_values,vmax=5,vmin=-5,xticklabels=False, yticklabels=False,cmap='coolwarm',ax=ax1)
    ax1.set_title('$\Delta\Delta$G wrt 11ntR (kcal/mol)')
    #plot ddG profiles as lines
    ax2 = fig.add_subplot(3,2,2); 
    (s1,s2) = next_cluster.shape
    cluster_title = ('cluster ' + str(counter + 1) + '   n = ' + str(s1))
    next_cluster_values.T.plot(ax=ax2,legend=False, title= cluster_title,marker='o',xticks=list(range(5,21,5)),grid=True,
                        color=color_list[counter + 1],ylim=(-4,4))
    #plot heatmap with dG values 
    ax3 = fig.add_subplot(3,2,3); 
    sns.heatmap(next_cluster_dG,vmax=-7,vmin=-12,yticklabels=False,xticklabels=False,ax=ax3)
    ax3.set_title('$\Delta$G (kcal/mol)')
    
    #plot dG profiles as lines
    ax4 = fig.add_subplot(3,2,4); 
    next_cluster_dG.T.plot(legend=False, marker='o',xticks=list(range(5,21,5)),grid=True,
                    color=color_list[counter + 1],ylim=(-12,0),ax = ax4)
    limitX = [1,20]
    limitY = [-7.1,-7.1]
    plt.plot(limitX,limitY,'k--',linewidth=3.0)
    
    #Create mask indicating values that were below stability threshold or NaN and plot heatmap. 
    ax5 = fig.add_subplot(3,2,5);  
    next_mask = next_cluster_dG <dG_threshold
    sns.heatmap(next_mask.astype('int64'),yticklabels=False, xticklabels=False,cmap='gray',ax=ax5)
    
    #Create bar plot indicating the number of receptors of each type. 
    ax6 = fig.add_subplot(3,2,6)
    number_types = [sum(next_cluster.receptor_type  == 1),sum(next_cluster.receptor_type  == 2),sum(next_cluster.receptor_type  == 3)]
    receptor_types = ['11ntR-like', 'in vitro', 'IC3/VC2-like']
    plt.bar([1,2,3],number_types,align='center')
    plt.ylim([0,250])
    plt.xticks([1,2,3],receptor_types)
    fig_name = 'cluster ' + str(counter + 1) + 'summary.pdf'

