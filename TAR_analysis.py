#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:39:34 2018

@author: Steve
"""

#%%
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
import itertools
#%%
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
TAR_Data = pd.read_csv(data_path + 'tectorna_results_tarjunctions.180122.csv')
#%%
plt.figure()
data_30mM = TAR_Data['dG_Mut2_GAAA']
plt.hist(data_30mM.dropna())

data_5mM = TAR_Data['dG_Mut2_GAAA_5mM_2']
plt.hist(data_5mM.dropna())

#%%
variants = set(TAR_Data['name'])
print(len(variants))












