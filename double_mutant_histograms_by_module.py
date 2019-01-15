#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:03:24 2018

@author: Steve
"""
'''--------------Import Libraries and functions--------------------'''
import pandas as pd
import numpy as np
#from clustering_functions import get_dG_for_scaffold
import seaborn as sns
import matplotlib.pyplot as plt
#%%
double_mutants_11ntR_all = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/double_mutants_11ntR_all_07_23_2018.pkl')
double_mutants_11ntR = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/double_mutants_11ntR_07_23_2018.pkl')
#%%
path_for_tables = '/Volumes/NO NAME/tables_11ntR_double_mutants'
#%% Mutation in Wobble and bulge
# THIS IS A SAMPLE; BLOCK BELOW DOES THIS FOR ALL OF SUBMODULE COMBINATIONS  (15 TOTAL)
#Create a table with the information for double mutants with mutations in specified
#submodules.
#So, for example the one below has mutations in the wobble and bulge submodules

#name for table and other plots in this block
name_table = 'wobble_bulge' 

#mask that selects only variants with mutations at specified submoduless
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.bulge_mutations
       
#pick specified variants; in this case I am taking all of the variants including
#those that have values above the threshold       
selected_variants = double_mutants_11ntR_all[mask]

#select only specified columns
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]
#change the name of the columnns so that they can be more easily visualized
selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]

#save table as spreadsheer
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')

#create histogram of coupling (observed - additive) for the variants specified 
#above
#where to save
path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'

#range and bins for histograms
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)

#plot different histogram for each of the five scaffolds
scaffolds = ['13854','14007','14073','35311','35600']

#save values from each scaffold into a common list to plot a histogram of all
#values at the end
all_scaffolds = []

for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    #select the coupling term for a specific scaffold
    A = selected_variants_coup['coupling__' + each_scaffold]
    #save values into common list to plot later
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()

#prepare list with all values for plotting histogram with information from 
#all five scaffolds
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)

#calculate statistics and plot them in histogram
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)

#textbox on histogram
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()
#%% CREATE TABLE, PLOT HISTOGRAMS FOR EACH 5 SCAFFOLDS, AND OVERALL HISTOGRAM 
#FOR EACH OF COMBINATION OF SUBMODULE DOUBLE MUTANTS
#Both Core
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.core_mutations
#------------------------       
name_table = 'both_core'
selected_variants = double_mutants_11ntR_all[mask]
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]

selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')


path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
scaffolds = ['13854','14007','14073','35311','35600']
all_scaffolds = []
for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    A = selected_variants_coup['coupling__' + each_scaffold]    
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    mu = A.mean()
    median = A.median()
    sigma = A.std()
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    plt.text(2, 3, textstr, fontsize=14,
        verticalalignment='top')
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()
  
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()       
#------------------------         
#both mutations in platform
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.platform_mutations
name_table = 'both_platform'
selected_variants = double_mutants_11ntR_all[mask]
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]

selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')


path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
scaffolds = ['13854','14007','14073','35311','35600']
all_scaffolds = []
for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    A = selected_variants_coup['coupling__' + each_scaffold]
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    mu = A.mean()
    median = A.median()
    sigma = A.std()
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    plt.text(2, 3, textstr, fontsize=14,
        verticalalignment='top')
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()
   
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()
#------------------------   
#both mutations in wobble
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.wobble_mutations
name_table = 'both_wobble'
selected_variants = double_mutants_11ntR_all[mask]
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]

selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')


path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
scaffolds = ['13854','14007','14073','35311','35600']
all_scaffolds = []
for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    A = selected_variants_coup['coupling__' + each_scaffold]
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    mu = A.mean()
    median = A.median()
    sigma = A.std()
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    plt.text(2, 3, textstr, fontsize=14,
        verticalalignment='top')
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()
   
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()
#------------------------   
#both mutations in bulge
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.bulge_mutations

#------------------------   
#both mutations in WC
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.WC_mutations
       
name_table = 'both_WC'
selected_variants = double_mutants_11ntR_all[mask]
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]

selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')


path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
scaffolds = ['13854','14007','14073','35311','35600']
all_scaffolds = []
for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    A = selected_variants_coup['coupling__' + each_scaffold]
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    mu = A.mean()
    median = A.median()
    sigma = A.std()
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    plt.text(2, 3, textstr, fontsize=14,
        verticalalignment='top')
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()
   
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()       
 #------------------------         
#Plot cases where both mutations happen in the different submodules

#mutations in core and platform 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.platform_mutations
       
name_table = 'core_platform'
selected_variants = double_mutants_11ntR_all[mask]
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]

selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')


path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
scaffolds = ['13854','14007','14073','35311','35600']
all_scaffolds = []
for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    A = selected_variants_coup['coupling__' + each_scaffold]
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    mu = A.mean()
    median = A.median()
    sigma = A.std()
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    plt.text(2, 3, textstr, fontsize=14,
        verticalalignment='top')
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()
   
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()       
       
#------------------------   
#mutations in core and wobble
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.wobble_mutations
name_table = 'core_wobble'
selected_variants = double_mutants_11ntR_all[mask]
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]

selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')


path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
scaffolds = ['13854','14007','14073','35311','35600']
all_scaffolds = []
for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    A = selected_variants_coup['coupling__' + each_scaffold]
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    mu = A.mean()
    median = A.median()
    sigma = A.std()
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    plt.text(2, 3, textstr, fontsize=14,
        verticalalignment='top')
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()
   
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()

#------------------------   
#mutations in core and bulge
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.bulge_mutations
name_table = 'core_bulge'
selected_variants = double_mutants_11ntR_all[mask]
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]

selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')


path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
scaffolds = ['13854','14007','14073','35311','35600']
all_scaffolds = []
for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    A = selected_variants_coup['coupling__' + each_scaffold]
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    mu = A.mean()
    median = A.median()
    sigma = A.std()
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    plt.text(2, 3, textstr, fontsize=14,
        verticalalignment='top')
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()
   
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()       
#------------------------         
#mutations in core and WC
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.WC_mutations
name_table = 'core_WC'
selected_variants = double_mutants_11ntR_all[mask]
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]

selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')


path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
scaffolds = ['13854','14007','14073','35311','35600']
all_scaffolds = []
for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    A = selected_variants_coup['coupling__' + each_scaffold]
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    mu = A.mean()
    median = A.median()
    sigma = A.std()
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    plt.text(2, 3, textstr, fontsize=14,
        verticalalignment='top')
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()
   
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()
#------------------------   
#mutations in wobble and platform 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.platform_mutations
name_table = 'wobble_platform'
selected_variants = double_mutants_11ntR_all[mask]
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]

selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')


path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
scaffolds = ['13854','14007','14073','35311','35600']
all_scaffolds = []
for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    A = selected_variants_coup['coupling__' + each_scaffold]
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    mu = A.mean()
    median = A.median()
    sigma = A.std()
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    plt.text(2, 3, textstr, fontsize=14,
        verticalalignment='top')
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()
   
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()              
#------------------------          
#mutations in wobble and bulge 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.bulge_mutations
name_table = 'wobble_bulge'
selected_variants = double_mutants_11ntR_all[mask]
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]

selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')


path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
scaffolds = ['13854','14007','14073','35311','35600']
all_scaffolds = []
for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    A = selected_variants_coup['coupling__' + each_scaffold]
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    mu = A.mean()
    median = A.median()
    sigma = A.std()
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    plt.text(2, 3, textstr, fontsize=14,
        verticalalignment='top')
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()
   
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()
#------------------------   
#mutations in wobble and WC 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.WC_mutations
name_table = 'wobble_WC'
selected_variants = double_mutants_11ntR_all[mask]
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]

selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')


path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
scaffolds = ['13854','14007','14073','35311','35600']
all_scaffolds = []
for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    A = selected_variants_coup['coupling__' + each_scaffold]
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    mu = A.mean()
    median = A.median()
    sigma = A.std()
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    plt.text(2, 3, textstr, fontsize=14,
        verticalalignment='top')
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()
   
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()
#------------------------   
#mutations in platform and bulge 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.platform_mutations &\
       double_mutants_11ntR.bulge_mutations
name_table = 'platform_bulge'
selected_variants = double_mutants_11ntR_all[mask]
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]

selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')


path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
scaffolds = ['13854','14007','14073','35311','35600']
all_scaffolds = []
for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    A = selected_variants_coup['coupling__' + each_scaffold]
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    mu = A.mean()
    median = A.median()
    sigma = A.std()
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    plt.text(2, 3, textstr, fontsize=14,
        verticalalignment='top')
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()
   
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()
#------------------------   
#mutations in platform and WC 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.platform_mutations &\
       double_mutants_11ntR.WC_mutations
name_table = 'platform_WC'
selected_variants = double_mutants_11ntR_all[mask]
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]

selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')


path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
scaffolds = ['13854','14007','14073','35311','35600']
all_scaffolds = []
for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    A = selected_variants_coup['coupling__' + each_scaffold]
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    mu = A.mean()
    median = A.median()
    sigma = A.std()
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    plt.text(2, 3, textstr, fontsize=14,
        verticalalignment='top')
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()
   
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()
#------------------------   
#mutations in bulge and WC 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.bulge_mutations &\
       double_mutants_11ntR.WC_mutations
name_table = 'bulge_WC'
selected_variants = double_mutants_11ntR_all[mask]
selected_variants_coup = selected_variants[['dG_30mM_Mg_GAAA_13854','ddG_additive_30mM_Mg_13854','coupling_30mM_Mg_13854',
                                 'dG_30mM_Mg_GAAA_14007','ddG_additive_30mM_Mg_14007','coupling_30mM_Mg_14007',
                                 'dG_30mM_Mg_GAAA_14073','ddG_additive_30mM_Mg_14073','coupling_30mM_Mg_14073',
                                 'dG_30mM_Mg_GAAA_35311','ddG_additive_30mM_Mg_35311','coupling_30mM_Mg_35311',
                                 'dG_30mM_Mg_GAAA_35600','ddG_additive_30mM_Mg_35600','coupling_30mM_Mg_35600',]]

selected_variants_coup.columns = ['ddG_obs_13854','ddG_add_13854','coupling__13854',
                             'ddG_obs_14007','ddG_add_14007','coupling__14007',
                             'ddG_obs_14073','ddG_add_14073','coupling__14073',
                             'ddG_obs_35311','ddG_add_35311','coupling__35311',
                             'ddG_obs_35600','ddG_add_35600','coupling__35600',]
selected_variants_coup.to_csv(path_for_tables + name_table +  '.csv')


path_for_hist = '/Volumes/NO NAME/hist_11ntR_double_mutants'
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
scaffolds = ['13854','14007','14073','35311','35600']
all_scaffolds = []
for each_scaffold in scaffolds:
    plt.figure()
    plt.title(name_table + '_' + each_scaffold)
    A = selected_variants_coup['coupling__' + each_scaffold]
    all_scaffolds = all_scaffolds + list(A.values)
    plt.hist(A.dropna(),range=R,bins = b)
    mu = A.mean()
    median = A.median()
    sigma = A.std()
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    plt.text(2, 3, textstr, fontsize=14,
        verticalalignment='top')
    figname = name_table + '_' + each_scaffold + '.svg'
    plt.savefig(path_for_hist + '/' + figname)
#    plt.show()
    plt.close()
   
all_scaffolds = [x for x in all_scaffolds if str(x) != 'nan']
all_scaffolds_array = np.asarray(all_scaffolds)
mu = all_scaffolds_array.mean()
median = np.median(all_scaffolds_array)
sigma = all_scaffolds_array.std()
plt.figure()
plt.title(name_table + '_all_scaffolds')
plt.hist(all_scaffolds_array,range=R,bins = b)
textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
plt.text(2, 15, textstr, fontsize=14,
        verticalalignment='top')
figname = name_table + '_all_scaffolds.svg'
plt.savefig(path_for_hist + '/' + figname)
#plt.show()
plt.close()

#%%-------------------PLOT HEATMAPS FOR EACH SUBMODULE VS SUBMODULE-------------------------------
#load WT data
#get data saved from other script
#sometimes it is necessary to provide the entire path
data_11ntR_scaffolds = pd.read_csv('data_11ntR_scaffolds.csv')
data_11ntR_scaffolds = data_11ntR_scaffolds.set_index('r_seq')
WT_data = data_11ntR_scaffolds.loc['UAUGG_CCUAAG'].copy()
#%% FOR EACH OF THE SUBMODULE COMBINATIONS CREATE 5 HEAT MAPS (ONE FOR EACH SCAFFOLD)
#WITH THE COUPLING VALUES 

#PLATFORM VS WC
#Do this for each of the five scaffolds.    
for each_scaffold in scaffolds:
    #platform vs WC
    #select variants with mutations at specified submodules
    mask = (double_mutants_11ntR.submodules_mutated == 2) &\
           double_mutants_11ntR.platform_mutations &\
           double_mutants_11ntR.WC_mutations
    selected_variants = double_mutants_11ntR_all[mask]
    
    #create dataframe where each of the variants will be assigned 2 indeces 
    #which corresponds to the x,y positions in the heatmap
    #this x,y positins will then be used to assign coupling values to a 
    #matrix that then will be used for plotting heatmap. 
    coordinates_select_var = pd.DataFrame(index = selected_variants.index, columns = ['index_1','index_2'])
    
    #The indices will be assigned based on the design of the heat map 
    #index 2 is for the horizontal axis
    #so for example, below, each of the mutations of the WC submodule will 
    #appear on the x axis of the heat map
    #and the mutations in the platform (index 1) will appear in the vertical 
    #axis of the heatmap.
    #GET COORDINATES FOR WC
    for variant in coordinates_select_var.index: 
        #  THIS IS FOR WC
        #We already know (because of mask) that these must have mutations in WC.
        #residue 1 in TLR corresponds to item 6 in index string
        #residue 11 in TLR corresponds to item 4 in index string
        if variant[4] == 'A':
            coordinates_select_var.loc[variant].index_2 = 3
        if variant[4] == 'C':
            coordinates_select_var.loc[variant].index_2 = 4   
        if variant[4] == 'U':
            coordinates_select_var.loc[variant].index_2 = 5
        # if variant[4]== 'G' then this is WT in that residue and mutation must be in variant[6]    
        if variant[6] == 'A':
            coordinates_select_var.loc[variant].index_2 = 0
        if variant[6] == 'G':
            coordinates_select_var.loc[variant].index_2 = 1
        if variant[6] == 'U':
            coordinates_select_var.loc[variant].index_2 = 2 
            
    #GET COORDINATES FOR PLATFORM        
    for variant in coordinates_select_var.index:
        # THIS IS FOR PLATFORM
        if variant[9] == 'G':
            coordinates_select_var.loc[variant].index_1 = 0      
        if variant[9] == 'C':
            coordinates_select_var.loc[variant].index_1 = 1  
        if variant[9] == 'U':
            coordinates_select_var.loc[variant].index_1 = 2          
        if variant[10] == 'G':
            coordinates_select_var.loc[variant].index_1 = 3      
        if variant[10] == 'C':
            coordinates_select_var.loc[variant].index_1 = 4  
        if variant[10] == 'U':
            coordinates_select_var.loc[variant].index_1 = 5      
            
    #CREATE MATRIX
    #this matrix will be filled with the coupling values based on the x,y positions
    #assigned above for each variant
    #in this case the matris is 6X6 because each submodule is 2 residues (WC and Platform)
    mutant_matrix = np.zeros((6,6))
    
    #FILL UP MATRIX
    for variant in coordinates_select_var.index:
        index1 = coordinates_select_var.loc[variant]['index_1']
        index2 = coordinates_select_var.loc[variant]['index_2']
        mutant_matrix[index1,index2] = selected_variants.loc[variant]['coupling_30mM_Mg_' + each_scaffold]  
    #PLOT HEATMAP
    plt.figure()
    plt.title('coupling_' + each_scaffold )
    sns.heatmap(mutant_matrix,vmax=4,vmin=-4,yticklabels=False,xticklabels=False,cmap='coolwarm',annot = True,linewidths=.5,linecolor = 'black')
    plt.show()
#%% PLOT HEAT MAP FOR COUPLING OF WOBBLE VS WC
name = 'wobble_WC_'
path_for_hist = '/Volumes/NO NAME/heatmaps_11ntR_double_mutants/'      
for each_scaffold in scaffolds:
    #calculate the threshold for each scaffold based on the WT value.
    #ddG_threshold depends on the dG of WT  
    ddG_thr = -7.1 - WT_data['dG_30mM_Mg_GAAA_' + each_scaffold]
    print(ddG_thr)
    
    #wobble vs WC
    mask = (double_mutants_11ntR.submodules_mutated == 2) &\
           double_mutants_11ntR.wobble_mutations &\
           double_mutants_11ntR.WC_mutations
    selected_variants = double_mutants_11ntR_all[mask]
    coordinates_select_var = pd.DataFrame(index = selected_variants.index, columns = ['index_1','index_2'])
    
    
    #GET COORDINATES FOR WC
    for variant in coordinates_select_var.index: 
        #  THIS IS FOR WC
        #We already know (because of mask) that these must have mutations in WC.
        #residue 1 in TLR corresponds to item 6 in index string
        #residue 11 in TLR corresponds to item 4 in index string
        if variant[4] == 'A':
            coordinates_select_var.loc[variant].index_2 = 3
        if variant[4] == 'C':
            coordinates_select_var.loc[variant].index_2 = 4   
        if variant[4] == 'U':
            coordinates_select_var.loc[variant].index_2 = 5
        # if variant[4]== 'G' then this is WT in that residue and mutation must be in variant[6]    
        if variant[6] == 'A':
            coordinates_select_var.loc[variant].index_2 = 0
        if variant[6] == 'G':
            coordinates_select_var.loc[variant].index_2 = 1
        if variant[6] == 'U':
            coordinates_select_var.loc[variant].index_2 = 2 
            
    #GET COORDINATES FOR WOBBLE        
    for variant in coordinates_select_var.index:
        if variant[11] == 'A':
            coordinates_select_var.loc[variant].index_1 = 0      
        if variant[11] == 'C':
            coordinates_select_var.loc[variant].index_1 = 1  
        if variant[11] == 'U':
            coordinates_select_var.loc[variant].index_1 = 2          
        if variant[0] == 'A':
            coordinates_select_var.loc[variant].index_1 = 3      
        if variant[0] == 'G':
            coordinates_select_var.loc[variant].index_1 = 4  
        if variant[0] == 'C':
            coordinates_select_var.loc[variant].index_1 = 5      
     
        
    #CREATE MATRIX
    mutant_matrix = np.zeros((6,6))
    
    #FILL UP MATRIX
    for variant in coordinates_select_var.index:
        index1 = coordinates_select_var.loc[variant]['index_1']
        index2 = coordinates_select_var.loc[variant]['index_2']
        mutant_matrix[index1,index2] = selected_variants.loc[variant]['coupling_30mM_Mg_' + each_scaffold]  
    #PLOT HEATMAP FOR COUPLING 
    plt.figure()
    plt.title('coupling_' + each_scaffold )
    sns.heatmap(mutant_matrix,vmax=4,vmin=-4,yticklabels=False,xticklabels=False,cmap='coolwarm',annot = True,linewidths=.5,linecolor = 'black')   
#    plt.savefig(path_for_hist + name + each_scaffold + '.svg') 
#    plt.close()  

    #Look for values that are above the ddG threshold\
    #if both the additive and the observed values are above ddG threshold place a 1
    #if the additive model is above the threshold place a 2
    #if the observed values is above the threshold place a 3
    mutant_matrix = np.zeros((6,6))
    for variant in coordinates_select_var.index:
        index1 = coordinates_select_var.loc[variant]['index_1']
        index2 = coordinates_select_var.loc[variant]['index_2']
        if selected_variants.loc[variant]['dG_30mM_Mg_GAAA_' + each_scaffold] > ddG_thr and\
        selected_variants.loc[variant]['ddG_additive_30mM_Mg_' + each_scaffold] > ddG_thr:
            mutant_matrix[index1,index2] = 1 
        elif selected_variants.loc[variant]['ddG_additive_30mM_Mg_' + each_scaffold] > ddG_thr:
            mutant_matrix[index1,index2] = 2 
        elif selected_variants.loc[variant]['dG_30mM_Mg_GAAA_' + each_scaffold] > ddG_thr:
            mutant_matrix[index1,index2] = 3
    plt.figure()    
    sns.heatmap(mutant_matrix,vmax=4,vmin=-4,yticklabels=False,xticklabels=False,cmap='coolwarm',annot = True,linewidths=.5,linecolor = 'black') 
#%% WOBBLE VS PLATFORM
name = 'wobble_platform_'
path_for_hist = '/Volumes/NO NAME/heatmaps_11ntR_double_mutants/'    
for each_scaffold in scaffolds:
    mask = (double_mutants_11ntR.submodules_mutated == 2) &\
           double_mutants_11ntR.wobble_mutations &\
           double_mutants_11ntR.platform_mutations
    selected_variants = double_mutants_11ntR_all[mask]
    coordinates_select_var = pd.DataFrame(index = selected_variants.index, columns = ['index_1','index_2'])
                
    #GET COORDINATES FOR WOBBLE        
    for variant in coordinates_select_var.index:
        if variant[11] == 'A':
            coordinates_select_var.loc[variant].index_1 = 0      
        if variant[11] == 'C':
            coordinates_select_var.loc[variant].index_1 = 1  
        if variant[11] == 'U':
            coordinates_select_var.loc[variant].index_1 = 2          
        if variant[0] == 'A':
            coordinates_select_var.loc[variant].index_1 = 3      
        if variant[0] == 'G':
            coordinates_select_var.loc[variant].index_1 = 4  
        if variant[0] == 'C':
            coordinates_select_var.loc[variant].index_1 = 5  
            
    #GET COORDINATES FOR PLATFORM        
    for variant in coordinates_select_var.index:
        # THIS IS FOR PLATFORM
        if variant[9] == 'G':
            coordinates_select_var.loc[variant].index_2 = 0      
        if variant[9] == 'C':
            coordinates_select_var.loc[variant].index_2 = 1  
        if variant[9] == 'U':
            coordinates_select_var.loc[variant].index_2 = 2          
        if variant[10] == 'G':
            coordinates_select_var.loc[variant].index_2 = 3      
        if variant[10] == 'C':
            coordinates_select_var.loc[variant].index_2 = 4  
        if variant[10] == 'U':
            coordinates_select_var.loc[variant].index_2 = 5             
    #CREATE MATRIX
    mutant_matrix = np.zeros((6,6))
    
    #FILL UP MATRIX
    for variant in coordinates_select_var.index:
        index1 = coordinates_select_var.loc[variant]['index_1']
        index2 = coordinates_select_var.loc[variant]['index_2']
        mutant_matrix[index1,index2] = selected_variants.loc[variant]['coupling_30mM_Mg_' + each_scaffold]  
    #PLOT HEATMAP
    plt.figure()
    plt.title('coupling_' + each_scaffold )
    sns.heatmap(mutant_matrix,vmax=4,vmin=-4,yticklabels=False,xticklabels=False,cmap='coolwarm',annot = True,linewidths=.5,linecolor = 'black')
    plt.savefig(path_for_hist + name + each_scaffold + '.svg') 
    plt.close()    
    
    
#%% This data will serve to create a heatmap describing the coupling between
# all residues in the TLR 
#11/12/2018
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'   
all_11ntR = pd.read_csv(data_path + 'all_11ntRs_unique.csv' )
# select those only with the normal flanking base pair
all_11ntR_normal = all_11ntR[all_11ntR['b_name'] == 'normal']
#select single point mutants. 
mask = ((all_11ntR.b_name == 'normal') & (all_11ntR.no_mutations == 1)) | ((all_11ntR.b_name == 'normal') & (all_11ntR.no_mutations == 0))
single_11ntR_mutants = all_11ntR[mask].copy()
#%
#group single point mutants by sequence
single_mutants_grouped = single_11ntR_mutants.groupby('r_seq')
#%
#select WT data 
WT = all_11ntR_normal[all_11ntR_normal['r_seq'] == 'UAUGG_CCUAAG'].copy()
WT = WT.set_index('old_idx')
#%
#make a list of the single point mutant sequences
list_single_mutants= list(set(single_11ntR_mutants['r_seq']))
list_single_mutants.remove('UAUGG_CCUAAG')
#%
single_mutant_ddG_summary = pd.DataFrame(index=list_single_mutants)
n_total_list = []
ddG_30Mg_list = []
n_30Mg = []

ddG_5Mg_list = []
n_5Mg = []

ddG_5Mg150K_list = []
n_5Mg150K = []

ddG_GUAA_list = []
n_GUAA = []

n_total = []

for sequences in list_single_mutants:
    next_data = single_mutants_grouped.get_group(sequences)
    n_total.append(len(next_data))
    next_data = next_data.set_index('old_idx')
    next_data = next_data.reindex(WT.index)
    
    ddG_30Mg = next_data['dG_Mut2_GAAA'] -  WT['dG_Mut2_GAAA']
    n_30Mg.append(len(ddG_30Mg.dropna()))
    
    ddG_5Mg = next_data['dG_Mut2_GAAA_5mM_2'] -  WT['dG_Mut2_GAAA_5mM_2']
    n_5Mg.append(len(ddG_5Mg.dropna()))
    
    ddG_5Mg150K = next_data['dG_Mut2_GAAA_5mM_150mMK_1'] -  WT['dG_Mut2_GAAA_5mM_150mMK_1']
    n_5Mg150K.append(len(ddG_5Mg150K.dropna()))
    
    ddG_GUAA = next_data['dG_Mut2_GUAA_1'] -  WT['dG_Mut2_GUAA_1']
    n_GUAA.append(len(ddG_GUAA.dropna()))
    
    
    ddG_30Mg_list.append(ddG_30Mg.median())
    ddG_5Mg_list.append(ddG_5Mg.median())
    ddG_5Mg150K_list.append(ddG_5Mg150K.median())
    ddG_GUAA_list.append(ddG_GUAA.median())
    
single_mutant_ddG_summary['ddG_30Mg'] = ddG_30Mg_list 
single_mutant_ddG_summary['n_30Mg'] = n_30Mg
 
single_mutant_ddG_summary['ddG_5Mg'] = ddG_5Mg_list
single_mutant_ddG_summary['n_5Mg'] = n_5Mg
   
single_mutant_ddG_summary['ddG_5Mg150K'] = ddG_5Mg150K_list 
single_mutant_ddG_summary['n_5Mg150K'] = n_5Mg150K
  
single_mutant_ddG_summary['ddG_GUAA'] = ddG_GUAA_list 
single_mutant_ddG_summary['n_GUAA'] = n_GUAA

single_mutant_ddG_summary['n_total_lib'] = n_total


#get mutation information from original dataframe
A = all_11ntR_normal.drop_duplicates(subset = 'r_seq').copy()
A = A.set_index('r_seq')
A = A.reindex(single_mutant_ddG_summary.index)
A = A[['mutations_ 1','mutations_ 2','mutations_ 3','mutations_ 4','mutations_ 5',
   'mutations_ 6','mutations_ 7','mutations_ 8','mutations_ 9','mutations_10',
   'mutations_11','mutations_12']]
single_mutant_ddG_summary = pd.concat([single_mutant_ddG_summary,A],axis = 1)

#get identity of mutation 
single_mutant_ddG_summary['mut_1_res'] = ''
single_mutant_ddG_summary['mut_2_res'] = ''
single_mutant_ddG_summary['mut_3_res'] = ''
single_mutant_ddG_summary['mut_4_res'] = ''
single_mutant_ddG_summary['mut_5_res'] = ''
single_mutant_ddG_summary['mut_6_res'] = ''
single_mutant_ddG_summary['mut_7_res'] = ''
single_mutant_ddG_summary['mut_8_res'] = ''
single_mutant_ddG_summary['mut_9_res'] = ''
single_mutant_ddG_summary['mut_10_res'] = ''
single_mutant_ddG_summary['mut_11_res'] = ''
single_mutant_ddG_summary['mut_12_res'] = ''

single_mutant_ddG_summary['sequence'] = single_mutant_ddG_summary.index

#for sequences in single_mutant_ddG_summary.index:
#    for counter in range(12):
#        if counter < 9:
#            label = 'mutations_ ' + str(counter + 1)
#        else:
#            label = 'mutations_' + str(counter + 1)
#        if single_mutant_ddG_summary.loc[sequences][label] == 1:
#            single_mutant_ddG_summary.loc[sequences]['mut_' + str(counter + 1) + '_res'] =\
#            sequences[counter]
#        else:
#            single_mutant_ddG_summary.loc[sequences]['mut_' + str(counter + 1) + '_res'] =\
#            '-'   
#%%
for counter in range (12):
    if counter < 9:
        label = 'mutations_ ' + str(counter + 1)
    else:
        label = 'mutations_' + str(counter + 1)  
    print(label)
    






         
#%% now do the same for double mutants




    