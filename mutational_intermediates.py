#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:18:12 2018

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
sublib5 = tecto_data[tecto_data['sublibrary'] == 'tertcontacts_5']
print('original size of sublibrary 5: ',sublib5.shape)
sublib5 = sublib5.drop_duplicates(subset='seq', keep="last")
print('after deleting duplicates: ',sublib5.shape)

#%%
unique_intermediates = sorted(list(set(sublib5.r_name)))

