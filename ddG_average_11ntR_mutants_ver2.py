#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 16:16:48 2018

@author: Steve
"""

#Important Note: The columns of all dataframes were kept as the original, i.e.
#with 'dG...' in the column label even when the values are ddGs. 


'''--------------Import Libraries and functions--------------------'''
import pandas as pd
import numpy as np
from clustering_functions import get_dG_for_scaffold
import seaborn as sns
import matplotlib.pyplot as plt
#import matplotlib as mpl
#from matplotlib.pyplot import *
#import scipy.cluster.hierarchy as sch
#import sklearn.decomposition as skd
#from scipy.cluster.hierarchy import cophenet
#from scipy.spatial.distance import pdist
#from scipy.cluster.hierarchy import fcluster
#from sklearn.neighbors import NearestNeighbors

apply_dG_threshold = 1 #For now values above threshold are replaced by NaN
replace_with = np.nan

'''---------------General Variables-------------'''
dG_threshold = -7.1 #kcal/mol; dG values above this are not reliable
dG_replace = -7.1 # for replacing values above threshold. 
nan_threshold = 0.50 #amount of missing data tolerated.
num_neighbors = 10
#for plotting
low_lim = -14
high_lim = -6

'''---------------Import data ------------------'''
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
#11ntR
all_11ntRs_unique_df = pd.read_csv(data_path + 'all_11ntRs_unique.csv')
unique_receptors_11ntR = sorted(list(set(list(all_11ntRs_unique_df.r_seq))))#used for indexing 
print(unique_receptors_11ntR[0])
print('# of unique 11ntR receptors: '+ str(len(unique_receptors_11ntR)))

#Create list of receptors in sublibrary and list of scaffolds
receptors_11ntR = sorted(list(set(all_11ntRs_unique_df['r_seq'])))
scaffolds_length = pd.concat([all_11ntRs_unique_df['old_idx'],all_11ntRs_unique_df['length']],axis=1)
scaffolds_length = scaffolds_length.drop_duplicates()
scaffolds_length = scaffolds_length.sort_values(by=['length'])
scaffolds_11ntR = list(scaffolds_length['old_idx'])
scaffolds_length = scaffolds_length.set_index('old_idx')
 #%%
#Note: in previous datatables (e.g. all_11ntR_unique.csv) old_idx imports as a 
# number.  However in the case of the original table 'tectorna_results_tertcontacts.180122.csv'
# old_idx imports as a string.
data = all_11ntRs_unique_df
conditions = ['dG_Mut2_GAAA']#, 'dG_Mut2_GUAA_1']
column_labels = ['dG_30mM_Mg_GAAA']#,'dG_30mM_Mg_GUAA']
row_index= ('r_seq')
flanking = 'normal'

data_11ntR_scaffolds_GAAA = pd.DataFrame(index = receptors_11ntR)

for scaffolds in scaffolds_11ntR:
    
    next_scaffold = scaffolds
    next_column_labels = [name + '_' +  str(next_scaffold) for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(receptors_11ntR)
    data_11ntR_scaffolds_GAAA = pd.concat([data_11ntR_scaffolds_GAAA,next_df],axis = 1)

conditions = ['dG_Mut2_GUAA_1']
column_labels = ['dG_30mM_Mg_GUAA']
row_index= ('r_seq')
flanking = 'normal'

data_11ntR_scaffolds_GUAA = pd.DataFrame(index = receptors_11ntR)

for scaffolds in scaffolds_11ntR:
    
    next_scaffold = scaffolds
    next_column_labels = [name + '_' +  str(next_scaffold) for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(receptors_11ntR)
    data_11ntR_scaffolds_GUAA = pd.concat([data_11ntR_scaffolds_GUAA,next_df],axis = 1)

conditions = ['dG_Mut2_GAAA_5mM_2']
column_labels = ['dG_5mM_Mg_GAAA']
row_index= ('r_seq')
flanking = 'normal'

data_11ntR_scaffolds_5Mg = pd.DataFrame(index = receptors_11ntR)

for scaffolds in scaffolds_11ntR:
    
    next_scaffold = scaffolds
    next_column_labels = [name + '_' +  str(next_scaffold) for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(receptors_11ntR)
    data_11ntR_scaffolds_5Mg = pd.concat([data_11ntR_scaffolds_5Mg,next_df],axis = 1)
    
conditions = ['dG_Mut2_GAAA_5mM_150mMK_1']
column_labels = ['dG_5Mg150K_GAAA']
row_index= ('r_seq')
flanking = 'normal'

data_11ntR_scaffolds_5Mg150K = pd.DataFrame(index = receptors_11ntR)

for scaffolds in scaffolds_11ntR:
    
    next_scaffold = scaffolds
    next_column_labels = [name + '_' +  str(next_scaffold) for name in column_labels]
    next_df = get_dG_for_scaffold(data,next_scaffold,conditions, 
                                              row_index, next_column_labels,flanking)
    next_df = next_df.reindex(receptors_11ntR)
    data_11ntR_scaffolds_5Mg150K = pd.concat([data_11ntR_scaffolds_5Mg150K,next_df],axis = 1)
   
data_11ntR_scaffolds = pd.concat([data_11ntR_scaffolds_GAAA,data_11ntR_scaffolds_GUAA,
                               data_11ntR_scaffolds_5Mg,
                               data_11ntR_scaffolds_5Mg150K],axis = 1)
#%% Create dataframe with mutations, residues mutated, and number of mutations. 
column_list = ['r_seq'] + list(all_11ntRs_unique_df.columns[-13 :])
mutations = all_11ntRs_unique_df[column_list].copy() 
mutations = mutations.drop_duplicates()
mutations = mutations.set_index('r_seq')
mutations = mutations.reindex(data_11ntR_scaffolds.index)
mut_1_res = []
mut_2_res = []
mut_3_res = []
mut_4_res = []
mut_5_res = []
mut_6_res = []
mut_7_res = []
mut_8_res = []
mut_9_res = []
mut_10_res = []
mut_11_res = []
mut_12_res = []
for sequences in mutations.index:
    if mutations.loc[sequences]['mutations_ 1'] == 1:
        mut_1_res.append(sequences[0])
    else:
        mut_1_res.append('-')
        
    if mutations.loc[sequences]['mutations_ 2'] == 1:
        mut_2_res.append(sequences[1])
    else:
        mut_2_res.append('-')

    if mutations.loc[sequences]['mutations_ 3'] == 1:
        mut_3_res.append(sequences[2])
    else:
        mut_3_res.append('-')        

    if mutations.loc[sequences]['mutations_ 4'] == 1:
        mut_4_res.append(sequences[3])
    else:
        mut_4_res.append('-')    
                
    if mutations.loc[sequences]['mutations_ 5'] == 1:
        mut_5_res.append(sequences[4])
    else:
        mut_5_res.append('-')
        
    if mutations.loc[sequences]['mutations_ 6'] == 1:
        mut_6_res.append(sequences[5])
    else:
        mut_6_res.append('-')
        
    if mutations.loc[sequences]['mutations_ 7'] == 1:
        mut_7_res.append(sequences[6])
    else:
        mut_7_res.append('-')
               
    if mutations.loc[sequences]['mutations_ 8'] == 1:
        mut_8_res.append(sequences[7])
    else:
        mut_8_res.append('-')        
        
    if mutations.loc[sequences]['mutations_ 9'] == 1:
        mut_9_res.append(sequences[8])
    else:
        mut_9_res.append('-')  
    
    if mutations.loc[sequences]['mutations_10'] == 1:
        mut_10_res.append(sequences[9])
    else:
        mut_10_res.append('-')

    if mutations.loc[sequences]['mutations_11'] == 1:
        mut_11_res.append(sequences[10])
    else:
        mut_11_res.append('-')      

    if mutations.loc[sequences]['mutations_12'] == 1:
        mut_12_res.append(sequences[11])
    else:
        mut_12_res.append('-')
        
mutations['mut_1_res'] = mut_1_res
mutations['mut_2_res'] = mut_2_res
mutations['mut_3_res'] = mut_3_res
mutations['mut_4_res'] = mut_4_res
mutations['mut_5_res'] = mut_5_res
mutations['mut_6_res'] = mut_6_res
mutations['mut_7_res'] = mut_7_res
mutations['mut_8_res'] = mut_8_res
mutations['mut_9_res'] = mut_9_res
mutations['mut_10_res'] = mut_10_res
mutations['mut_11_res'] = mut_11_res
mutations['mut_12_res'] = mut_12_res
#%%
data_11ntR_scaffolds.to_csv('data_11ntR_scaffolds.csv')
#%%Create a mask for values above threshold and keep it for future reference
dG_threshold_mask = data_11ntR_scaffolds > dG_threshold
#data_11ntR_scaffolds_threshold = data_11ntR_scaffolds.copy()
#if apply_dG_threshold == 1:
#    data_11ntR_scaffolds_threshold[dG_threshold_mask] = np.nan
#%% Normalized data relative to the canonical 11ntR
WT_data = data_11ntR_scaffolds.loc['UAUGG_CCUAAG'].copy()
#%% Create ddG matrix by subtracting WT from every column
ddG_11ntR_mutants = data_11ntR_scaffolds.sub(WT_data)
ddG_11ntR_mutants_threshold = ddG_11ntR_mutants.copy()
if apply_dG_threshold == 1:
    ddG_11ntR_mutants_threshold[dG_threshold_mask] = replace_with
sns.heatmap(ddG_11ntR_mutants_threshold)
#%% take only ddG values for 30 mM Mg with GAAA tetraloop
ddG_11ntR_mutants_30Mg_GAAA = ddG_11ntR_mutants_threshold[ddG_11ntR_mutants.columns[0:50]].copy()
#%% Calculate ddG_average
ddG_average_30Mg_GAAA = ddG_11ntR_mutants_30Mg_GAAA.mean(axis = 1)
ddG_std_30Mg_GAAA = ddG_11ntR_mutants_30Mg_GAAA.std(axis = 1)
# Append mutation information
ddG_11ntR_mutants_30Mg_GAAA = pd.concat([ddG_11ntR_mutants_30Mg_GAAA,mutations],axis=1)
#%%Append the ddG average and standard deviation
ddG_11ntR_mutants_30Mg_GAAA['ddG_average'] = ddG_average_30Mg_GAAA
ddG_11ntR_mutants_30Mg_GAAA['ddG_std'] = ddG_std_30Mg_GAAA
#%%Separate single and double mutants 
single_mutants_11ntR = ddG_11ntR_mutants_30Mg_GAAA[ddG_11ntR_mutants_30Mg_GAAA.no_mutations == 1].copy()
double_mutants_11ntR = ddG_11ntR_mutants_30Mg_GAAA[ddG_11ntR_mutants_30Mg_GAAA.no_mutations == 2].copy()
#%%Calculate predicted ddG based on additive model using single mutant effects
#create an empty list to append the cumulative ddG_additive as looping over 
#every row of the double mutants
ddG_additive = []

#divide the dataframes to include only the rows in the mutation information and
#the ddG average values
column_list2 = list(double_mutants_11ntR.columns[63:])
sM_matrix = single_mutants_11ntR[column_list2].copy()
dM_matrix = double_mutants_11ntR[column_list2].copy()
dM_matrix['ddG_additive'] = 0

#loop over every row in the double mutant dataframe and every time that a mutation 
#is foung (i.e. the cell != '-') then look for that specific residue in the single
#mutant dataframe; after found it can break the loop
for row_dM in dM_matrix.index:
    ddG_counter = 0
    if dM_matrix.loc[row_dM]['mut_1_res'] != '-':
        residue = dM_matrix.loc[row_dM]['mut_1_res']
        for row_sM in sM_matrix.index:
            if sM_matrix.loc[row_sM]['mut_1_res'] == residue:
                ddG_counter = ddG_counter + sM_matrix.loc[row_sM]['ddG_average']
                break

    if dM_matrix.loc[row_dM]['mut_2_res'] != '-':
        residue = dM_matrix.loc[row_dM]['mut_2_res']
        for row_sM in sM_matrix.index:
            if sM_matrix.loc[row_sM]['mut_2_res'] == residue:
                ddG_counter = ddG_counter + sM_matrix.loc[row_sM]['ddG_average']
                break

    if dM_matrix.loc[row_dM]['mut_3_res'] != '-':
        residue = dM_matrix.loc[row_dM]['mut_3_res']
        for row_sM in sM_matrix.index:
            if sM_matrix.loc[row_sM]['mut_3_res'] == residue:
                ddG_counter = ddG_counter + sM_matrix.loc[row_sM]['ddG_average']
                break

    if dM_matrix.loc[row_dM]['mut_4_res'] != '-':
        residue = dM_matrix.loc[row_dM]['mut_4_res']
        for row_sM in sM_matrix.index:
            if sM_matrix.loc[row_sM]['mut_4_res'] == residue:
                ddG_counter = ddG_counter + sM_matrix.loc[row_sM]['ddG_average']
                break

    if dM_matrix.loc[row_dM]['mut_5_res'] != '-':
        residue = dM_matrix.loc[row_dM]['mut_5_res']
        for row_sM in sM_matrix.index:
            if sM_matrix.loc[row_sM]['mut_5_res'] == residue:
                ddG_counter = ddG_counter + sM_matrix.loc[row_sM]['ddG_average']
                break

    if dM_matrix.loc[row_dM]['mut_6_res'] != '-':
        residue = dM_matrix.loc[row_dM]['mut_6_res']
        for row_sM in sM_matrix.index:
            if sM_matrix.loc[row_sM]['mut_6_res'] == residue:
                ddG_counter = ddG_counter + sM_matrix.loc[row_sM]['ddG_average']
                break

    if dM_matrix.loc[row_dM]['mut_7_res'] != '-':
        residue = dM_matrix.loc[row_dM]['mut_7_res']
        for row_sM in sM_matrix.index:
            if sM_matrix.loc[row_sM]['mut_7_res'] == residue:
                ddG_counter = ddG_counter + sM_matrix.loc[row_sM]['ddG_average']
                break

    if dM_matrix.loc[row_dM]['mut_8_res'] != '-':
        residue = dM_matrix.loc[row_dM]['mut_8_res']
        for row_sM in sM_matrix.index:
            if sM_matrix.loc[row_sM]['mut_8_res'] == residue:
                ddG_counter = ddG_counter + sM_matrix.loc[row_sM]['ddG_average']
                break

    if dM_matrix.loc[row_dM]['mut_9_res'] != '-':
        residue = dM_matrix.loc[row_dM]['mut_9_res']
        for row_sM in sM_matrix.index:
            if sM_matrix.loc[row_sM]['mut_9_res'] == residue:
                ddG_counter = ddG_counter + sM_matrix.loc[row_sM]['ddG_average']
                break
            
    if dM_matrix.loc[row_dM]['mut_10_res'] != '-':
        residue = dM_matrix.loc[row_dM]['mut_10_res']
        for row_sM in sM_matrix.index:
            if sM_matrix.loc[row_sM]['mut_10_res'] == residue:
                ddG_counter = ddG_counter + sM_matrix.loc[row_sM]['ddG_average']
                break

    if dM_matrix.loc[row_dM]['mut_11_res'] != '-':
        residue = dM_matrix.loc[row_dM]['mut_11_res']
        for row_sM in sM_matrix.index:
            if sM_matrix.loc[row_sM]['mut_11_res'] == residue:
                ddG_counter = ddG_counter + sM_matrix.loc[row_sM]['ddG_average']
                break

    if dM_matrix.loc[row_dM]['mut_12_res'] != '-':
        residue = dM_matrix.loc[row_dM]['mut_12_res']
        for row_sM in sM_matrix.index:
            if sM_matrix.loc[row_sM]['mut_12_res'] == residue:
                ddG_counter = ddG_counter + sM_matrix.loc[row_sM]['ddG_average']
                break

    ddG_additive.append(ddG_counter) 
#%% Append prediction for additive contribution from single mutations
double_mutants_11ntR['ddG_additive'] = ddG_additive
#%% Save dataframes
double_mutants_11ntR.to_csv('double_mutants_11ntR_ddG.csv')
single_mutants_11ntR.to_csv('single_mutants_11ntR_ddG.csv')

#%% Plot as scatter plot the predicted from additive model vs the observer ddG
plt.scatter(double_mutants_11ntR.ddG_average,double_mutants_11ntR.ddG_additive,
            edgecolors='k')
xlim = [-1,6]
ddG_threshold = [3.8,3.8]
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot(xlim,ddG_threshold,'--k',linewidth = 1.5)
plt.plot(ddG_threshold,xlim,'--k',linewidth = 1.5)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#%% Create columns identifyin which of the submodules were mutated in each of the 
#double mutants. 
core_mutations = (double_mutants_11ntR['mutations_ 2'] == 1) |\
                 (double_mutants_11ntR['mutations_ 4'] == 1) |\
                 (double_mutants_11ntR['mutations_ 8'] == 1) |\
                 (double_mutants_11ntR['mutations_ 9'] == 1)
platform_mutations = (double_mutants_11ntR['mutations_10'] == 1) |\
                     (double_mutants_11ntR['mutations_11'] == 1)
wobble_mutations = (double_mutants_11ntR['mutations_ 1'] == 1) |\
                     (double_mutants_11ntR['mutations_12'] == 1) 
bulge_mutations = (double_mutants_11ntR['mutations_ 3'] == 1)
WC_mutations = (double_mutants_11ntR['mutations_ 5'] == 1) |\
                     (double_mutants_11ntR['mutations_ 7'] == 1) 
                                       
double_mutants_11ntR['core_mutations'] = core_mutations
double_mutants_11ntR['platform_mutations'] = platform_mutations
double_mutants_11ntR['wobble_mutations'] = wobble_mutations
double_mutants_11ntR['bulge_mutations'] = bulge_mutations
double_mutants_11ntR['WC_mutations'] = WC_mutations

double_mutants_11ntR['submodules_mutated'] = (double_mutants_11ntR.core_mutations).astype(int) +\
                                             (double_mutants_11ntR.platform_mutations).astype(int) +\
                                             (double_mutants_11ntR.wobble_mutations).astype(int) +\
                                             (double_mutants_11ntR.bulge_mutations).astype(int) +\
                                             (double_mutants_11ntR.WC_mutations).astype(int)

#%%
#Scatter plots for ddG_average (i.e. the average accross scaffolds)                                           

#Create scatter plots showing ddG_obs vs. ddG_additive for each 'type' of double
#mutant. For example, double mutants with both mutations in the wobble 'submodule'
#or one mutation in the wobble and another in the AA-platform.                                              

#select line width for double mutants 'rings'
LW = 3                                       
                                             
#Plot cases where both mutations happen in the same submodules
#both mutations in core
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.core_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='k',s=120,c='yellow',linewidth=0.25)  
#both mutations in platform
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.platform_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='k',s=120,c='green',linewidth=0.25)  
#both mutations in wobble
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.wobble_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='k',s=120,c='purple',linewidth=0.25) 
#both mutations in bulge
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.bulge_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='k',s=120,c='orange',linewidth=0.25) 
#both mutations in WC
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='k',s=120,c='red',linewidth=0.25) 

#Plot cases where both mutations happen in the different submodules

#mutations in core and platform 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.platform_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='green',s=120,c='yellow',linewidth=LW) 
#mutations in core and wobble
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.wobble_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='purple',s=120,c='yellow',linewidth=LW) 
#mutations in core and bulge
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.bulge_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='orange',s=120,c='yellow',linewidth=LW) 
#mutations in core and WC
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='red',s=120,c='yellow',linewidth=LW) 


#mutations in wobble and platform 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.platform_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='green',s=120,c='purple',linewidth=LW) 
#mutations in wobble and bulge 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.bulge_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='orange',s=120,c='purple',linewidth=LW)
#mutations in wobble and WC 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='red',s=120,c='purple',linewidth=LW)



#mutations in platform and bulge 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.platform_mutations &\
       double_mutants_11ntR.bulge_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='orange',s=120,c='green',linewidth=LW) 
#mutations in platform and WC 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.platform_mutations &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='red',s=120,c='green',linewidth=LW) 
#mutations in bulge and WC 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.bulge_mutations &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR[mask].ddG_average,
            double_mutants_11ntR[mask].ddG_additive,
            edgecolors='red',s=120,c='orange',linewidth=LW) 
                
xlim = [-1,6]
ddG_threshold = [3.8,3.8]
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot(xlim,ddG_threshold,'--k',linewidth = 1.5)
plt.plot(ddG_threshold,xlim,'--k',linewidth = 1.5)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)

#%% Histograms for ddG_average (i.e. ddG across all the scaffolds)
#Create histogram for each 'type' of double mutant; for example, when both muta-
#tions occur in the wobble 'submodule' or when one mutation is in the wobble and
#the other is in the AA-platform

#The histograms report on the coupling energy, based on the difference between
#the observed ddG for the double mutant and that predicted from an additive 
#model (i.e. sum of effects of single mutants)
plot_histograms = 0

if plot_histograms == 1:

    R=[-4, 4] #range
    bw = 0.5 #binwidth
    b = np.arange(R[0], R[1] + bw, bw)
    plt.title('both mutation in core') 
    mask = (double_mutants_11ntR.submodules_mutated == 1) &\
           double_mutants_11ntR.core_mutations
    A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
    plt.hist(A.dropna(),range=R,bins = b)
    plt.show() 
    
    plt.figure()
    plt.title('both mutations in platform')
    mask = (double_mutants_11ntR.submodules_mutated == 1) &\
           double_mutants_11ntR.platform_mutations
    A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
    plt.hist(A.dropna(),range=R,bins = b)
    plt.show()
    
    plt.figure()
    plt.title('both mutations in wobble')
    mask = (double_mutants_11ntR.submodules_mutated == 1) &\
           double_mutants_11ntR.wobble_mutations
    A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
    plt.hist(A.dropna(),range=R,bins = b)
    plt.show()
    
    plt.figure()
    plt.title('both mutations in WC')
    mask = (double_mutants_11ntR.submodules_mutated == 1) &\
           double_mutants_11ntR.WC_mutations
    A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
    plt.hist(A.dropna(),range=R,bins = b)
    plt.show()
    
    
    plt.figure()
    plt.title('mutations in core and platform')
    mask = (double_mutants_11ntR.submodules_mutated == 2) &\
           double_mutants_11ntR.core_mutations &\
           double_mutants_11ntR.platform_mutations
    A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
    plt.hist(A.dropna(),range=R,bins = b)
    plt.show()
           
    plt.figure()
    plt.title('mutations in core and wobble')
    mask = (double_mutants_11ntR.submodules_mutated == 2) &\
           double_mutants_11ntR.core_mutations &\
           double_mutants_11ntR.wobble_mutations
    A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
    plt.hist(A.dropna(),range=R,bins = b)
    plt.show()
           
    plt.figure()
    plt.title('mutations in core and bulge')
    mask = (double_mutants_11ntR.submodules_mutated == 2) &\
           double_mutants_11ntR.core_mutations &\
           double_mutants_11ntR.bulge_mutations
    A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
    plt.hist(A.dropna(),range=R,bins = b)
    plt.show()       
           
    plt.figure()
    plt.title('mutations in core and WC')
    mask = (double_mutants_11ntR.submodules_mutated == 2) &\
           double_mutants_11ntR.core_mutations &\
           double_mutants_11ntR.WC_mutations
    A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
    plt.hist(A.dropna(),range=R,bins = b)
    plt.show()       
    
    
    plt.figure()
    plt.title('mutations in wobble and platform') 
    mask = (double_mutants_11ntR.submodules_mutated == 2) &\
           double_mutants_11ntR.wobble_mutations &\
           double_mutants_11ntR.platform_mutations
    A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
    plt.hist(A.dropna(),range=R,bins = b)
    plt.show()       
    
    
    plt.figure()
    plt.title('mutations in wobble and bulge') 
    mask = (double_mutants_11ntR.submodules_mutated == 2) &\
           double_mutants_11ntR.wobble_mutations &\
           double_mutants_11ntR.bulge_mutations
    A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
    plt.hist(A.dropna(),range=R,bins = b)
    plt.show()       
    
    
    plt.figure()
    plt.title('mutations in wobble and WC')
    mask = (double_mutants_11ntR.submodules_mutated == 2) &\
           double_mutants_11ntR.wobble_mutations &\
           double_mutants_11ntR.WC_mutations
    A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
    plt.hist(A.dropna(),range=R,bins = b)
    plt.show()       
    
    plt.figure()
    plt.title('mutations in platform and bulge') 
    mask = (double_mutants_11ntR.submodules_mutated == 2) &\
           double_mutants_11ntR.platform_mutations &\
           double_mutants_11ntR.bulge_mutations
    A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
    plt.hist(A.dropna(),range=R,bins = b)
    plt.show()  
    
    plt.figure()
    plt.title('mutations in platform and WC')
    mask = (double_mutants_11ntR.submodules_mutated == 2) &\
           double_mutants_11ntR.platform_mutations &\
           double_mutants_11ntR.WC_mutations
    A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
    plt.hist(A.dropna(),range=R,bins = b)
    plt.show()  
    
    plt.figure()
    plt.title('mutations in bulge and WC')
    mask = (double_mutants_11ntR.submodules_mutated == 2) &\
           double_mutants_11ntR.bulge_mutations &\
           double_mutants_11ntR.WC_mutations
    A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
    plt.hist(A.dropna(),range=R,bins = b)
    plt.show()  

#%%Calculate additive ddG for each scaffold 

#select the scaffold and sum the effects of each single mutant within that 
#that scaffold. 

scaffolds = ['35600', '35311', '14073', '14007', '13854']
#These next lines are just taking the information about the sequence of the 
#mutations for each single and double mutant.  These information was generated 
#above.
column_list2 = list(double_mutants_11ntR.columns[63:75])
sM_matrix = single_mutants_11ntR[column_list2].copy()
dM_matrix = double_mutants_11ntR[column_list2].copy()

for each_scaff in scaffolds:

    #Change these two names to calculate the additive ddG predicted for a double 
    #mutant based on the effects on single mutants. 
    data_column = 'dG_30mM_Mg_GAAA_' + each_scaff
    new_col_to_append = 'ddG_additive_30mM_Mg_' + each_scaff
    
    #Calculate predicted ddG based on additive model using single mutant effects
    #create an empty list to append the cumulative ddG_additive as looping over 
    #every row of the double mutants
    ddG_additive = []
    
    #divide the dataframes to include only the rows in the mutation information and
    #the ddG average values

    
    #loop over every row in the double mutant dataframe and every time that a mutation 
    #is found (i.e. the cell != '-') then look for that specific residue in the single
    #mutant dataframe take that effect (ddG) and add it; after found it can break the loop
    
    #Important Note: The columns of all dataframes were kept as the original, i.e.
    #with 'dG...' in the column label even when the values are ddGs. 
    
    for row_dM in dM_matrix.index:
        ddG_counter = 0
        if dM_matrix.loc[row_dM]['mut_1_res'] != '-':
            residue = dM_matrix.loc[row_dM]['mut_1_res']
            for row_sM in sM_matrix.index:
                if sM_matrix.loc[row_sM]['mut_1_res'] == residue:
                    ddG_counter = ddG_counter + single_mutants_11ntR.loc[row_sM][data_column]
                    break
    
        if dM_matrix.loc[row_dM]['mut_2_res'] != '-':
            residue = dM_matrix.loc[row_dM]['mut_2_res']
            for row_sM in sM_matrix.index:
                if sM_matrix.loc[row_sM]['mut_2_res'] == residue:
                    ddG_counter = ddG_counter + single_mutants_11ntR.loc[row_sM][data_column]
                    break
    
        if dM_matrix.loc[row_dM]['mut_3_res'] != '-':
            residue = dM_matrix.loc[row_dM]['mut_3_res']
            for row_sM in sM_matrix.index:
                if sM_matrix.loc[row_sM]['mut_3_res'] == residue:
                    ddG_counter = ddG_counter + single_mutants_11ntR.loc[row_sM][data_column]
                    break
    
        if dM_matrix.loc[row_dM]['mut_4_res'] != '-':
            residue = dM_matrix.loc[row_dM]['mut_4_res']
            for row_sM in sM_matrix.index:
                if sM_matrix.loc[row_sM]['mut_4_res'] == residue:
                    ddG_counter = ddG_counter + single_mutants_11ntR.loc[row_sM][data_column]
                    break
    
        if dM_matrix.loc[row_dM]['mut_5_res'] != '-':
            residue = dM_matrix.loc[row_dM]['mut_5_res']
            for row_sM in sM_matrix.index:
                if sM_matrix.loc[row_sM]['mut_5_res'] == residue:
                    ddG_counter = ddG_counter + single_mutants_11ntR.loc[row_sM][data_column]
                    break
    
        if dM_matrix.loc[row_dM]['mut_6_res'] != '-':
            residue = dM_matrix.loc[row_dM]['mut_6_res']
            for row_sM in sM_matrix.index:
                if sM_matrix.loc[row_sM]['mut_6_res'] == residue:
                    ddG_counter = ddG_counter + single_mutants_11ntR.loc[row_sM][data_column]
                    break
    
        if dM_matrix.loc[row_dM]['mut_7_res'] != '-':
            residue = dM_matrix.loc[row_dM]['mut_7_res']
            for row_sM in sM_matrix.index:
                if sM_matrix.loc[row_sM]['mut_7_res'] == residue:
                    ddG_counter = ddG_counter + single_mutants_11ntR.loc[row_sM][data_column]
                    break
    
        if dM_matrix.loc[row_dM]['mut_8_res'] != '-':
            residue = dM_matrix.loc[row_dM]['mut_8_res']
            for row_sM in sM_matrix.index:
                if sM_matrix.loc[row_sM]['mut_8_res'] == residue:
                    ddG_counter = ddG_counter + single_mutants_11ntR.loc[row_sM][data_column]
                    break
    
        if dM_matrix.loc[row_dM]['mut_9_res'] != '-':
            residue = dM_matrix.loc[row_dM]['mut_9_res']
            for row_sM in sM_matrix.index:
                if sM_matrix.loc[row_sM]['mut_9_res'] == residue:
                    ddG_counter = ddG_counter + single_mutants_11ntR.loc[row_sM][data_column]
                    break
                
        if dM_matrix.loc[row_dM]['mut_10_res'] != '-':
            residue = dM_matrix.loc[row_dM]['mut_10_res']
            for row_sM in sM_matrix.index:
                if sM_matrix.loc[row_sM]['mut_10_res'] == residue:
                    ddG_counter = ddG_counter + single_mutants_11ntR.loc[row_sM][data_column]
                    break
    
        if dM_matrix.loc[row_dM]['mut_11_res'] != '-':
            residue = dM_matrix.loc[row_dM]['mut_11_res']
            for row_sM in sM_matrix.index:
                if sM_matrix.loc[row_sM]['mut_11_res'] == residue:
                    ddG_counter = ddG_counter + single_mutants_11ntR.loc[row_sM][data_column]
                    break
    
        if dM_matrix.loc[row_dM]['mut_12_res'] != '-':
            residue = dM_matrix.loc[row_dM]['mut_12_res']
            for row_sM in sM_matrix.index:
                if sM_matrix.loc[row_sM]['mut_12_res'] == residue:
                    ddG_counter = ddG_counter + single_mutants_11ntR.loc[row_sM][data_column]
                    break
    
        ddG_additive.append(ddG_counter) 
    # Append prediction for additive contribution from single mutations
    double_mutants_11ntR[new_col_to_append] = ddG_additive
#%% Scatter plot for ddG_obs vs. ddG_additive for each scaffold
put_back_thr = 1

if put_back_thr == 1:
    all_ddG_30mM_Mg = ddG_11ntR_mutants[list(ddG_11ntR_mutants.columns[0:50])]
    double_mutants_11ntR_all = all_ddG_30mM_Mg.reindex(double_mutants_11ntR.index)
    additional_columns = double_mutants_11ntR[list(double_mutants_11ntR.columns[50:])]
    double_mutants_11ntR_all = pd.concat([double_mutants_11ntR_all,additional_columns],axis=1)
else:
    double_mutants_11ntR_all = double_mutants_11ntR.copy()
#%%
#Change the next two lines based on the scaffold that needs to be plotted
observed = 'dG_30mM_Mg_GAAA_35311'
additive = 'ddG_additive_30mM_Mg_35311'

#ddG_threshold depends on the dG of WT  


ddG_thr = -7.1 - WT_data[observed]

#select line width for double mutants 'rings'
LW = 3                                       
                                             
#Plot cases where both mutations happen in the same submodules
#both mutations in core
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.core_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='k',s=120,c='yellow',linewidth=0.25)  

#both mutations in platform
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.platform_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='k',s=120,c='green',linewidth=0.25)  
#both mutations in wobble
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.wobble_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='k',s=120,c='purple',linewidth=0.25) 
#both mutations in bulge
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.bulge_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='k',s=120,c='orange',linewidth=0.25) 
#both mutations in WC
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='k',s=120,c='red',linewidth=0.25) 



#Plot cases where both mutations happen in the different submodules
#mutations in core and platform 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.platform_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='green',s=120,c='yellow',linewidth=LW) 
#mutations in core and wobble
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.wobble_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='purple',s=120,c='yellow',linewidth=LW) 
#mutations in core and bulge
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.bulge_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='orange',s=120,c='yellow',linewidth=LW) 
#mutations in core and WC
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='red',s=120,c='yellow',linewidth=LW) 


#mutations in wobble and platform 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.platform_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='green',s=120,c='purple',linewidth=LW) 
#mutations in wobble and bulge 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.bulge_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='orange',s=120,c='purple',linewidth=LW)
#mutations in wobble and WC 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='red',s=120,c='purple',linewidth=LW)



#mutations in platform and bulge 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.platform_mutations &\
       double_mutants_11ntR.bulge_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='orange',s=120,c='green',linewidth=LW) 
#mutations in platform and WC 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.platform_mutations &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='red',s=120,c='green',linewidth=LW) 
#mutations in bulge and WC 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.bulge_mutations &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='red',s=120,c='orange',linewidth=LW) 
                
xlim = [-1,7]
ddG_threshold = [ddG_thr,ddG_thr]
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
#plt.plot(ddG_threshold,xlim,'--k',linewidth = 1.5)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)


#%%
R=[-4, 4] #range
bw = 0.5 #binwidth
b = np.arange(R[0], R[1] + bw, bw)
plt.title('both mutation in core') 
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.core_mutations
coupling_35600 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35600 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35600      
coupling_35311 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35311 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35311        
coupling_14073 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14073 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14073                 
coupling_14007 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14007 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14007 
coupling_13854 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_13854 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_13854 
A = pd.concat([coupling_35600,coupling_35311,coupling_14073,coupling_14007,coupling_13854])
#A =  double_mutants_11ntR[mask].ddG_average - double_mutants_11ntR[mask].ddG_additive      
plt.hist(A.dropna(),range=R,bins = b)
plt.show()


plt.figure()
plt.title('both mutations in platform')
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.platform_mutations
coupling_35600 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35600 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35600      
coupling_35311 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35311 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35311        
coupling_14073 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14073 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14073                 
coupling_14007 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14007 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14007 
coupling_13854 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_13854 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_13854 
A = pd.concat([coupling_35600,coupling_35311,coupling_14073,coupling_14007,coupling_13854])     
plt.hist(A.dropna(),range=R,bins = b)
plt.show()

plt.figure()
plt.title('both mutations in wobble')
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.wobble_mutations
coupling_35600 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35600 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35600      
coupling_35311 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35311 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35311        
coupling_14073 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14073 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14073                 
coupling_14007 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14007 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14007 
coupling_13854 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_13854 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_13854 
A = pd.concat([coupling_35600,coupling_35311,coupling_14073,coupling_14007,coupling_13854])     
plt.hist(A.dropna(),range=R,bins = b)
plt.show()

plt.figure()
plt.title('both mutations in WC')
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.WC_mutations
coupling_35600 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35600 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35600      
coupling_35311 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35311 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35311        
coupling_14073 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14073 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14073                 
coupling_14007 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14007 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14007 
coupling_13854 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_13854 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_13854 
A = pd.concat([coupling_35600,coupling_35311,coupling_14073,coupling_14007,coupling_13854])     
plt.hist(A.dropna(),range=R,bins = b)
plt.show()


plt.figure()
plt.title('mutations in core and platform')
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.platform_mutations
coupling_35600 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35600 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35600      
coupling_35311 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35311 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35311        
coupling_14073 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14073 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14073                 
coupling_14007 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14007 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14007 
coupling_13854 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_13854 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_13854 
A = pd.concat([coupling_35600,coupling_35311,coupling_14073,coupling_14007,coupling_13854])     
plt.hist(A.dropna(),range=R,bins = b)
plt.show()
       
plt.figure()
plt.title('mutations in core and wobble')
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.wobble_mutations
coupling_35600 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35600 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35600      
coupling_35311 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35311 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35311        
coupling_14073 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14073 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14073                 
coupling_14007 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14007 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14007 
coupling_13854 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_13854 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_13854 
A = pd.concat([coupling_35600,coupling_35311,coupling_14073,coupling_14007,coupling_13854])      
plt.hist(A.dropna(),range=R,bins = b)
plt.show()
       
plt.figure()
plt.title('mutations in core and bulge')
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.bulge_mutations
coupling_35600 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35600 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35600      
coupling_35311 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35311 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35311        
coupling_14073 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14073 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14073                 
coupling_14007 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14007 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14007 
coupling_13854 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_13854 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_13854 
A = pd.concat([coupling_35600,coupling_35311,coupling_14073,coupling_14007,coupling_13854])      
plt.hist(A.dropna(),range=R,bins = b)
plt.show()       
       
plt.figure()
plt.title('mutations in core and WC')
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.WC_mutations
coupling_35600 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35600 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35600      
coupling_35311 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35311 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35311        
coupling_14073 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14073 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14073                 
coupling_14007 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14007 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14007 
coupling_13854 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_13854 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_13854 
A = pd.concat([coupling_35600,coupling_35311,coupling_14073,coupling_14007,coupling_13854])      
plt.hist(A.dropna(),range=R,bins = b)
plt.show()       


plt.figure()
plt.title('mutations in wobble and platform') 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.platform_mutations
coupling_35600 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35600 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35600      
coupling_35311 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35311 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35311        
coupling_14073 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14073 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14073                 
coupling_14007 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14007 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14007 
coupling_13854 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_13854 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_13854 
A = pd.concat([coupling_35600,coupling_35311,coupling_14073,coupling_14007,coupling_13854])      
plt.hist(A.dropna(),range=R,bins = b)
plt.show()       


plt.figure()
plt.title('mutations in wobble and bulge') 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.bulge_mutations
coupling_35600 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35600 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35600      
coupling_35311 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35311 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35311        
coupling_14073 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14073 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14073                 
coupling_14007 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14007 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14007 
coupling_13854 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_13854 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_13854 
A = pd.concat([coupling_35600,coupling_35311,coupling_14073,coupling_14007,coupling_13854])      
plt.hist(A.dropna(),range=R,bins = b)
plt.show()       


plt.figure()
plt.title('mutations in wobble and WC')
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.WC_mutations
coupling_35600 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35600 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35600      
coupling_35311 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35311 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35311        
coupling_14073 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14073 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14073                 
coupling_14007 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14007 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14007 
coupling_13854 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_13854 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_13854 
A = pd.concat([coupling_35600,coupling_35311,coupling_14073,coupling_14007,coupling_13854])      
plt.hist(A.dropna(),range=R,bins = b)
plt.show()       

plt.figure()
plt.title('mutations in platform and bulge') 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.platform_mutations &\
       double_mutants_11ntR.bulge_mutations
coupling_35600 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35600 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35600      
coupling_35311 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35311 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35311        
coupling_14073 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14073 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14073                 
coupling_14007 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14007 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14007 
coupling_13854 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_13854 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_13854 
A = pd.concat([coupling_35600,coupling_35311,coupling_14073,coupling_14007,coupling_13854])      
plt.hist(A.dropna(),range=R,bins = b)
plt.show()  

plt.figure()
plt.title('mutations in platform and WC')
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.platform_mutations &\
       double_mutants_11ntR.WC_mutations
coupling_35600 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35600 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35600      
coupling_35311 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35311 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35311        
coupling_14073 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14073 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14073                 
coupling_14007 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14007 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14007 
coupling_13854 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_13854 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_13854 
A = pd.concat([coupling_35600,coupling_35311,coupling_14073,coupling_14007,coupling_13854])      
plt.hist(A.dropna(),range=R,bins = b)
plt.show()  

plt.figure()
plt.title('mutations in bulge and WC')
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.bulge_mutations &\
       double_mutants_11ntR.WC_mutations
coupling_35600 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35600 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35600      
coupling_35311 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_35311 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_35311        
coupling_14073 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14073 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14073                 
coupling_14007 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_14007 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_14007 
coupling_13854 = double_mutants_11ntR[mask].dG_30mM_Mg_GAAA_13854 - double_mutants_11ntR[mask].ddG_additive_30mM_Mg_13854 
A = pd.concat([coupling_35600,coupling_35311,coupling_14073,coupling_14007,coupling_13854])      
plt.hist(A.dropna(),range=R,bins = b)
plt.show()  


#%% PLOT INDIVIDUAL SCATTER PLOTS.
#Change the next two lines based on the scaffold that needs to be plotted

path_for_figures = ('/Volumes/NO NAME/TectoRNA_TL_TLR/ddG_11ntR_double_mutants_5scaffolds_individually/')
observed = 'dG_30mM_Mg_GAAA_13854'
additive = 'ddG_additive_30mM_Mg_13854'

#ddG_threshold depends on the dG of WT  
ddG_thr = -7.1 - WT_data[observed]
ddG_threshold = [ddG_thr,ddG_thr]
#select line width for double mutants 'rings'
LW = 3                                       

xlim = [-1,8]                                           
#Plot cases where both mutations happen in the same submodules
plt.figure()
plt.title('both mutations in core_' + observed[-5:])
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.core_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='k',s=120,c='yellow',linewidth=0.25)  

plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show()
plt.savefig(path_for_figures + 'fig_1_' + observed[-5:] + '.svg')
plt.close()

plt.figure()
plt.title('both mutations in platform' + observed[-5:]) 
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.platform_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='k',s=120,c='green',linewidth=0.25)

plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show()
plt.savefig(path_for_figures + 'fig_2_' + observed[-5:] + '.svg')
plt.close()

plt.figure()  
plt.title('both mutations in wobble_' + observed[-5:])
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.wobble_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='k',s=120,c='purple',linewidth=0.25) 
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show()
plt.savefig(path_for_figures + 'fig_3_' + observed[-5:] + '.svg')
plt.close()

plt.figure()
plt.title('both mutations in bulge_' + observed[-5:])
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.bulge_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='k',s=120,c='orange',linewidth=0.25) 
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show()
plt.savefig(path_for_figures + 'fig_4_' + observed[-5:] + '.svg')
plt.close()

plt.figure()
plt.title('both mutations in WC_' +  observed[-5:])
mask = (double_mutants_11ntR.submodules_mutated == 1) &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='k',s=120,c='red',linewidth=0.25) 
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show()
plt.savefig(path_for_figures + 'fig_5_' + observed[-5:] + '.svg')
plt.close()
#Plot cases where both mutations happen in the different submodules

plt.figure()
plt.title('mutations in core and platform _' + observed[-5:])
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.platform_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='green',s=120,c='yellow',linewidth=LW)
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show()
plt.savefig(path_for_figures + 'fig_6_' + observed[-5:] + '.svg')
plt.close()

plt.figure() 
plt.title('mutations in core and wobble_' + observed[-5:])
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.wobble_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='purple',s=120,c='yellow',linewidth=LW) 
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show()
plt.savefig(path_for_figures + 'fig_7_' + observed[-5:] + '.svg')
plt.close()

plt.figure()
plt.title('mutations in core and bulge_' + observed[-5:])
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.bulge_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='orange',s=120,c='yellow',linewidth=LW)
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show() 
plt.savefig(path_for_figures + 'fig_8_' + observed[-5:] + '.svg')
plt.close()

plt.figure()
plt.title('mutations in core and WC_' + observed[-5:])
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.core_mutations &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='red',s=120,c='yellow',linewidth=LW) 
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show()
plt.savefig(path_for_figures + 'fig_9_' + observed[-5:] + '.svg')
plt.close()

plt.figure()
plt.title('mutations in wobble and platform_' + observed[-5:])
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.platform_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='green',s=120,c='purple',linewidth=LW)
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show()
plt.savefig(path_for_figures + 'fig_10_' + observed[-5:] + '.svg')
plt.close()

plt.figure() 
plt.title('mutations in wobble and bulge_' + observed[-5:]) 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.bulge_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='orange',s=120,c='purple',linewidth=LW)
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show()
plt.savefig(path_for_figures + 'fig_11_' + observed[-5:] + '.svg')
plt.close()

plt.figure()
plt.title('mutations in wobble and WC_' + observed[-5:])
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='red',s=120,c='purple',linewidth=LW)
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show()
plt.savefig(path_for_figures + 'fig_12_' + observed[-5:] + '.svg')
plt.close()

plt.figure()
plt.title('mutations in platform and bulge_' + observed[-5:]) 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.platform_mutations &\
       double_mutants_11ntR.bulge_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='orange',s=120,c='green',linewidth=LW) 
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show()
plt.savefig(path_for_figures + 'fig_13_' + observed[-5:] + '.svg')
plt.close()

plt.figure()
plt.title('mutations in platform and WC_' + observed[-5:]) 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.platform_mutations &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='red',s=120,c='green',linewidth=LW) 
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show()
plt.savefig(path_for_figures + 'fig_14_' + observed[-5:] + '.svg')
plt.close()

plt.figure()
plt.title('mutations in bulge and WC_' + observed[-5:]) 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.bulge_mutations &\
       double_mutants_11ntR.WC_mutations
plt.scatter(double_mutants_11ntR_all[mask][observed],
            double_mutants_11ntR_all[mask][additive],
            edgecolors='red',s=120,c='orange',linewidth=LW) 
plt.plot(xlim,xlim,'--r',linewidth = 3)
plt.plot([ddG_thr,xlim[1]],ddG_threshold,'--k',linewidth = 3)
plt.plot(ddG_threshold,[ddG_thr,xlim[1]],'--k',linewidth = 3)
plt.xlim(xlim)
plt.ylim(xlim)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta\Delta$G$_{obs}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta\Delta$G$_{additive}$ (kcal/mol)',fontsize=22)
#plt.show()
plt.savefig(path_for_figures + 'fig_15_' + observed[-5:] + '.svg')
plt.close()

#%%
scaffolds = ['35600', '35311', '14073', '14007', '13854']
#These next lines are just taking the information about the sequence of the 
#mutations for each single and double mutant.  These information was generated 
for each_scaff in scaffolds:

    #Change these two names to calculate the additive ddG predicted for a double 
    #mutant based on the effects on single mutants. 
    observed_label = 'dG_30mM_Mg_GAAA_' + each_scaff
    additive_label = 'ddG_additive_30mM_Mg_' + each_scaff
    coupling_label = 'coupling_30mM_Mg_' + each_scaff
    double_mutants_11ntR_all[coupling_label] = double_mutants_11ntR_all[observed_label] - double_mutants_11ntR_all[additive_label]
#%%    
#mutations in wobble and platform 
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.wobble_mutations &\
       double_mutants_11ntR.WC_mutations
coupling_df = double_mutants_11ntR_all[mask][['coupling_30mM_Mg_13854','coupling_30mM_Mg_14007','coupling_30mM_Mg_14073','coupling_30mM_Mg_35311','coupling_30mM_Mg_35600']]       



#%%
mask = (double_mutants_11ntR.submodules_mutated == 2) &\
       double_mutants_11ntR.platform_mutations &\
       double_mutants_11ntR.WC_mutations

x = double_mutants_11ntR[mask]['dG_30mM_Mg_GAAA_35311']
y = double_mutants_11ntR[mask]['ddG_additive_30mM_Mg_35311']
plt.scatter(x,y)
coupling = x - y
plt.figure()
plt.hist(coupling.dropna())  
plt.show()     





