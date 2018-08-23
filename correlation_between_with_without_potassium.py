#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:35:38 2018

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

'''--------------Import Functions--------------''' 
from clustering_functions import get_dG_for_scaffold
from clustering_functions import doPCA
from clustering_functions import interpolate_mat_knn
from clustering_functions import prep_data_for_clustering_ver2

#General Variables
dG_threshold = -6.5 #kcal/mol; dG values above this are not reliable
dG_replace = -6.5 # for replacing values above threshold. 
nan_threshold = 1 #amount of missing data tolerated.
num_neighbors = 10 # for interpolation

#import data analyzed previously
data_5_scaffolds = pd.read_pickle('prep_data_all_TLRs_5_scaffolds_with_nan.pkl')
receptors_analyzed = data_5_scaffolds.index
#%%
#import data from csv files 
#Data has been separated into 11ntR, IC3, and in vitro
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
entire_lib = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
#Drop repeats, just because they are there; is the same data
entire_lib = entire_lib.drop_duplicates(subset ='seq')
#Consider the ones only with normal closing base pair 
mask = entire_lib.b_name == 'normal' 
entire_lib_normal_bp = entire_lib[mask]
#Exclude sublibrary 5 which has the mutation intermediates and cannot be easily classified (or maybe they can be classified as others)
mask = (entire_lib_normal_bp.sublibrary == 'tertcontacts_0') | (entire_lib_normal_bp.sublibrary == 'tertcontacts_1')|\
       (entire_lib_normal_bp.sublibrary == 'tertcontacts_2') | (entire_lib_normal_bp.sublibrary == 'tertcontacts_3')|\
       (entire_lib_normal_bp.sublibrary == 'tertcontacts_4') | (entire_lib_normal_bp.sublibrary == 'tertcontacts_5')
entire_lib_selected = entire_lib_normal_bp[mask].copy()
#Create new name identifier because there are some that were given the same name
#but they are different receptors
#also attacht the sublibrary they came from for later reference
entire_lib_selected['new_name'] = entire_lib_selected.r_name + '_' + entire_lib_selected.r_seq + '_' + entire_lib_selected.sublibrary
#%%Calculate pearson coefficients between 5 and 5 + potassium
#HAVE TO REMOVE VALUES THAT ARE LIMITS
potassium_correlation = pd.DataFrame(index=receptors_analyzed)
potassium_correlation_list = []
n_list = []
low_lim = -14
high_lim = -6
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
for receptors in potassium_correlation.index:
    library = entire_lib_selected.copy()
    receptor_data = entire_lib_selected[library.new_name == receptors]
    data_5mM_receptor = receptor_data.dG_Mut2_GAAA_5mM_2.copy()
    data_5mM_receptor[data_5mM_receptor> dG_threshold] = np.nan
    data_5mM150K_receptor = receptor_data.dG_Mut2_GAAA_5mM_150mMK_1.copy()
    data_5mM150K_receptor[data_5mM150K_receptor> dG_threshold] = np.nan
    data_comb = pd.concat([data_5mM_receptor,data_5mM150K_receptor],axis = 1)
    R = data_comb.corr(method = 'pearson')
    R_diag = R.dG_Mut2_GAAA_5mM_150mMK_1.dG_Mut2_GAAA_5mM_2
    potassium_correlation_list.append(R_diag)
    #calculate n
    s = (data_5mM150K_receptor.isna()) | (data_5mM_receptor.isna())
    a = ~s
    n = a.sum()
    n_list.append(n)   
#    if n > 2 and R_diag < 0.75:
#        plt.figure()
#        plt.scatter(data_5mM_receptor,data_5mM150K_receptor,s=120,edgecolors='k',marker ='s')
#        plt.plot(x,x,':k')
#        plt.plot(x,y_thres,':k',linewidth = 0.5)
#        plt.plot(y_thres,x,':k',linewidth = 0.5)
#        plt.xlim(low_lim,high_lim)
#        plt.ylim(low_lim,high_lim)
#        plt.xticks(list(range(-14,-4,2)))
#        plt.yticks(list(range(-14,-4,2)))
#        plt.tick_params(axis='both', which='major', labelsize=24)
#        plt.axes().set_aspect('equal')
#        plt.xlabel('$\Delta$G$^{11ntR}_{bind}$ (kcal/mol)',fontsize=22)
#        plt.ylabel('$\Delta$G$^{mut}_{bind}$ (kcal/mol)',fontsize=22)
#        plt.title(str(receptors))
potassium_correlation['pearson_corr'] = potassium_correlation_list
potassium_correlation['n'] = n_list
#%%
n_threshold = 3
subset = potassium_correlation[potassium_correlation.n>n_threshold]
plt.hist(subset.pearson_corr.dropna())
subset_receptors = subset.index
#how many subplots in each direction
x_subplots = 3
y_subplots = 3

low_lim = -14
high_lim = -6
x = [low_lim, dG_threshold] #for plotting  y= x line
counter = -1
fig, axs = plt.subplots(x_subplots, y_subplots, sharey=True, sharex = True, tight_layout=True)
axs = axs.ravel()

figure_counter = 0
for receptors in subset.index:
    counter += 1
    if counter < (x_subplots * y_subplots):
        library = entire_lib_selected.copy()
        receptor_data = entire_lib_selected[library.new_name == receptors]
        R_coeff = subset.loc[receptors]['pearson_corr']
        name = receptor_data.r_name.iloc[0] + '_' +  receptor_data.r_seq.iloc[0]
        data_5mM_receptor = receptor_data.dG_Mut2_GAAA_5mM_2.copy()
        data_5mM150K_receptor = receptor_data.dG_Mut2_GAAA_5mM_150mMK_1.copy()
        axs[counter].scatter(data_5mM_receptor,data_5mM150K_receptor,edgecolor = 'black')
        axs[counter].plot(x,x,':k')
        axs[counter].plot(x,y_thres,':k',linewidth = 0.5)
        axs[counter].plot(y_thres,x,':k',linewidth = 0.5)
        axs[counter].set_xlim(low_lim,high_lim)
        axs[counter].set_ylim(low_lim,high_lim)
        axs[counter].set_xticks(list(range(-14,-4,2)))
        axs[counter].set_yticks(list(range(-14,-4,2)))
        axs[counter].tick_params(axis='both', which='major')
        axs[counter].set_aspect('equal')
        #axs[0].set_xlabel('$\Delta$G$^{11ntR}_{bind}$ (kcal/mol)',fontsize=22)
        #axs[0].set_ylabel('$\Delta$G$^{mut}_{bind}$ (kcal/mol)',fontsize=22)
        axs[counter].set_title(str(name),fontsize=6)
        #calculate averga ddG (exclude values below threshold)
        data_5mM_receptor[data_5mM_receptor > dG_threshold] = np.nan
        data_5mM150K_receptor[data_5mM150K_receptor > dG_threshold] = np.nan
        ddG = data_5mM150K_receptor.subtract(data_5mM_receptor)
        ddG_average = ddG.mean()
        x_ddG = [ddG_average + x[0],ddG_average + x[1]]
        axs[counter].plot(x,x_ddG,'--r',linewidth = 3)
        textstr = '$\Delta\Delta G=%.2f$\n$\mathrm{r}=%.2f$' % (ddG_average, R_coeff)
        axs[counter].text(-9, -12, textstr, fontsize=7,
        verticalalignment='top')
    else:
        figure_counter += 1
#        fig.savefig('/Volumes/NO NAME/potassium_vs_none/with_without_K_' + str(figure_counter) + '.pdf')
        fig, axs = plt.subplots(x_subplots, y_subplots, sharey=True, sharex = True, tight_layout=True)
        axs = axs.ravel()
        library = entire_lib_selected.copy()
        receptor_data = entire_lib_selected[library.new_name == receptors]
        R_coeff = subset.loc[receptors]['pearson_corr']
        name = receptor_data.r_name.iloc[0] + '_' +  receptor_data.r_seq.iloc[0]
        data_5mM_receptor = receptor_data.dG_Mut2_GAAA_5mM_2.copy()
        data_5mM150K_receptor = receptor_data.dG_Mut2_GAAA_5mM_150mMK_1.copy()
        axs[0].scatter(data_5mM_receptor,data_5mM150K_receptor,edgecolor = 'black')
        axs[0].plot(x,x,':k')
        axs[0].plot(x,y_thres,':k',linewidth = 0.5)
        axs[0].plot(y_thres,x,':k',linewidth = 0.5)
        axs[0].set_xlim(low_lim,high_lim)
        axs[0].set_ylim(low_lim,high_lim)
        axs[0].set_xticks(list(range(-14,-4,2)))
        axs[0].set_yticks(list(range(-14,-4,2)))
        axs[0].tick_params(axis='both', which='major')
        axs[0].set_aspect('equal')
        #axs[0].set_xlabel('$\Delta$G$^{11ntR}_{bind}$ (kcal/mol)',fontsize=22)
        #axs[0].set_ylabel('$\Delta$G$^{mut}_{bind}$ (kcal/mol)',fontsize=22)
        axs[0].set_title(str(name),fontsize=6)
        #calculate averga ddG (exclude values below threshold)
        data_5mM_receptor[data_5mM_receptor > dG_threshold] = np.nan
        data_5mM150K_receptor[data_5mM150K_receptor > dG_threshold] = np.nan
        ddG = data_5mM150K_receptor.subtract(data_5mM_receptor)
        ddG_average = ddG.mean()
        x_ddG = [ddG_average + x[0],ddG_average + x[1]]
        axs[0].plot(x,x_ddG,'--r',linewidth = 3)
        textstr = '$\Delta\Delta G=%.2f$\n$\mathrm{r}=%.2f$' % (ddG_average, R_coeff)
        axs[0].text(-9, -12, textstr, fontsize=7,
        verticalalignment='top')        
        counter = 0
        
figure_counter += 1
#fig.savefig('/Volumes/NO NAME/potassium_vs_none/with_without_K_' + str(figure_counter) + '.pdf')
