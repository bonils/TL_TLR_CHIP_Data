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
'''--------------Import Functions----------------------''' 
from Cluster_5_scaffolds import get_dG_for_scaffold
from Cluster_5_scaffolds import doPCA
from Cluster_5_scaffolds import interpolate_mat_knn
from Cluster_5_scaffolds import prep_data_for_clustering_ver2
#%%
#General Variables
dG_threshold = -7.1 #kcal/mol; dG values above this are not reliable
dG_replace = -7.1 # for replacing values above threshold. 
nan_threshold = 0.50 #amount of missing data tolerated.
#%%

