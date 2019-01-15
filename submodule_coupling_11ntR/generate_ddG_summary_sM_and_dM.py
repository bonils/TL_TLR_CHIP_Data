#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:46:01 2018

@author: Steve
"""
#This script generates data summaries with ddG across ALL scaffolds available
#relative to 11ntR wt.
#The median was used as the average. 
#The summaries also include 0s and 1s for residues that were mutated
#and also the identity of the mutation.
#THE SUMMARIES ARE SAVED AS TWO SEPARATE PKL FILES, ONE FOR SINGLE MUTANTS
#AND ONE FOR DOUBLE MUTANTS. 

'''--------------Import Libraries and functions--------------------'''
import pandas as pd
#from clustering_functions import get_dG_for_scaffold
double_mutants_11ntR_all = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/double_mutants_11ntR_all_07_23_2018.pkl')
double_mutants_11ntR = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/double_mutants_11ntR_07_23_2018.pkl')
#%%
#import information about receptors that have been selected previously for analysis
receptors_types_matrix = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/'
                        'initial_clustering_all_data/receptors_types_matrix.pkl')

#from double mutant data previously generated select only those being analyzed further
selected_11ntRs = receptors_types_matrix[receptors_types_matrix['type_11ntR'] == 1]
selected = []
for sequence in double_mutants_11ntR_all.index:
    if sequence in selected_11ntRs.index:
        selected.append(1)
    else:
        selected.append(0)

double_mutants_11ntR['selected'] = selected
double_mutants_11ntR_all['selected'] = selected

double_mutants_11ntR = double_mutants_11ntR[double_mutants_11ntR['selected'] == 1]
double_mutants_11ntR_all = double_mutants_11ntR_all[double_mutants_11ntR_all['selected'] == 1]
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

#generate single point mutant ddG summary dataframe
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
single_mutant_ddG_summary['mut_1_res'] = '-'
single_mutant_ddG_summary['mut_2_res'] = '-'
single_mutant_ddG_summary['mut_3_res'] = '-'
single_mutant_ddG_summary['mut_4_res'] = '-'
single_mutant_ddG_summary['mut_5_res'] = '-'
single_mutant_ddG_summary['mut_6_res'] = '-'
single_mutant_ddG_summary['mut_7_res'] = '-'
single_mutant_ddG_summary['mut_8_res'] = '-'
single_mutant_ddG_summary['mut_9_res'] = '-'
single_mutant_ddG_summary['mut_10_res'] = '-'
single_mutant_ddG_summary['mut_11_res'] = '-'
single_mutant_ddG_summary['mut_12_res'] = '-'

single_mutant_ddG_summary['sequence'] = single_mutant_ddG_summary.index
WT_seq = 'UAUGG_CCUAAG'
for sequence in single_mutant_ddG_summary.index:
    for index in range(len(WT_seq)):
        if sequence[index] != WT_seq[index]:
            single_mutant_ddG_summary['mut_' + str(index + 1) + '_res'].loc[sequence] =\
            sequence[index]
#%%
#create similar data summary for double mutants. 
double_mutant_ddG_summary = pd.DataFrame(index = double_mutants_11ntR_all.index)
all_11ntR_grouped = all_11ntR_normal.groupby('r_seq')

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

for sequences in double_mutant_ddG_summary.index:
    next_data = all_11ntR_grouped.get_group(sequences)
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
    
double_mutant_ddG_summary['ddG_30Mg'] = ddG_30Mg_list 
double_mutant_ddG_summary['n_30Mg'] = n_30Mg
 
double_mutant_ddG_summary['ddG_5Mg'] = ddG_5Mg_list
double_mutant_ddG_summary['n_5Mg'] = n_5Mg
   
double_mutant_ddG_summary['ddG_5Mg150K'] = ddG_5Mg150K_list 
double_mutant_ddG_summary['n_5Mg150K'] = n_5Mg150K
  
double_mutant_ddG_summary['ddG_GUAA'] = ddG_GUAA_list 
double_mutant_ddG_summary['n_GUAA'] = n_GUAA

double_mutant_ddG_summary['n_total_lib'] = n_total
            
A = double_mutants_11ntR_all[double_mutants_11ntR_all.columns[50:75]]
B = double_mutants_11ntR_all[double_mutants_11ntR_all.columns[78:84]]
double_mutant_ddG_summary = pd.concat([double_mutant_ddG_summary,A,B],axis = 1)      
#%%
#save data 
single_mutant_ddG_summary.to_pickle('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/submodule_coupling_11ntR/single_mutant_ddG_summary.pkl')
double_mutant_ddG_summary.to_pickle('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/submodule_coupling_11ntR/double_mutant_ddG_summary.pkl')        
        
        
        
        