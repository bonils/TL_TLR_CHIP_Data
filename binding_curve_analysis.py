#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:04:57 2018

@author: Steve
"""

# This script visualizes binding curves
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
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
binding_data = pd.read_csv(data_path + 'Mut2_GAAA_1.PerVariant.CPseries',
                           delim_whitespace = True)
binding_data = binding_data.set_index('variant_number')

#variant_ids = pd.read_csv(data_path + 'tecto_undetermined.CPannot.CPannot',
#                           delim_whitespace = True)
tecto_data = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')

#%%
variant = 27062
fluorescence = binding_data.loc[variant]
concentration = pd.Series([0.91e-9,2.74e-9,8.23e-9,24.7e-9,74.1e-9,222e-9,667e-9,2e-6])
plt.scatter(concentration,fluorescence)
plt.xlim(0,2000e-9)
