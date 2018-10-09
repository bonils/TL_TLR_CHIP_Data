#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:14:17 2018

@author: Steve
"""
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
data = pd.read_pickle('prep_data_all_TLRs_5_scaffolds_with_nan.pkl')
#%%
min_num_to_average=1
max_num_to_average=20
threshold=0.2
enforce_threshold=False
metric='cityblock'
#%%

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
#%%    
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
