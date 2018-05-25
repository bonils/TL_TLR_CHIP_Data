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
    data = data.copy()
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


def prep_data_for_clustering_ver2 (data,dG_thr, dG_rep, max_nan,num_neighbors):
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
            B,info = interpolate_mat_knn(B,max_num_to_average=num_neighbors)
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

