#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:08:37 2018

@author: Steve
"""
#%%
#import libraries
import pandas as pd
import numpy as np
#%%
#import previously analyzed data 
double_mutant_ddG_summary = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/'
                                          'TL_TLR_CHIP_Data/submodule_coupling_11ntR/'
                                          'double_mutant_ddG_summary_pred_3.pkl') 

single_mutant_ddG_summary = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/'
                                          'TL_TLR_CHIP_Data/submodule_coupling_11ntR/'
                                          'single_mutant_ddG_summary.pkl')
#%%
#function for comparing each character of two strings a la Matlab
def compare_strings (a,b):
    #they have to have the same length
    compared_list = []
    for pos in range(len(a)):
        if a[pos] == b[pos]:
            compared_list.append(1)
        else:
            compared_list.append(0)
    return compared_list

#%%
def gen_double_mutants (seq):  
    mutation = compare_strings(seq,'UAUGG_CCUAAG')
    residues = ['A','C','G','U']
    seq = list(seq)
    list_mutants = []
    for i in range(len(mutation)):
        if (mutation[i] == 1) & (seq[i] != '_'):
            for each in residues:
                next_mutant = seq.copy()
                if each != next_mutant[i]:
                    next_mutant[i] = each
                    list_mutants.append("".join(next_mutant))
    return list(set(list_mutants))
#%%
# put single point mutants in diagonal of array
matrix = np.empty([49,49],dtype = object)
matrix[:] = ''

wt_sequence = 'UAUGG_CCUAAG'
dummy_list = ['UCUGG_CCUAAG']

#if wt is A
for_A = ['C','G','U']
for_C = ['A','G','U']
for_G = ['A','C','U']
for_U = ['A','C','G']


for i in single_mutant_ddG_summary.index:
    sequence = i
    mutation = compare_strings(i,wt_sequence)
    for j in range(len(mutation)):
        if mutation[j] == 0:
            #choose the residues that need to be mutated
            wt_list = list(wt_sequence)
            if wt_list[j] == 'A':
                nucleotides = for_A
            elif wt_list[j] == 'C':
                nucleotides = for_C
            elif wt_list[j] == 'G':
                nucleotides = for_G
            elif wt_list[j] == 'U':
                nucleotides = for_U

            string = list(sequence)
            residue = string[j]
            
            if residue == nucleotides[0]:
                idx_X = (j) * 3 + 0
            if residue == nucleotides[1]:
                idx_X = (j) * 3 + 1
            if residue == nucleotides[2]:
                idx_X = (j) * 3 + 2
#            if residue == 'U':
#                idx_X = (j) * 4 + 3
    idx_Y = idx_X
    print(idx_X)
    print(sequence)
    matrix[idx_X,idx_Y] = sequence

#%%
#generate all possible double mutants
single_mutants = list(single_mutant_ddG_summary.index) 
double_mutants = []
for singles in single_mutants:
    double_mutants.append(gen_double_mutants(singles))

double_mutants = [each for sublist in double_mutants for each in sublist]
double_mutants = list(set(double_mutants))

#verify that they are all double mutants
number_mutations = []
for each in double_mutants:
    mutation = compare_strings(each,'UAUGG_CCUAAG')
    number_mutations.append(12 - np.array(mutation).sum())
#%%
# locate double mutants
#dummy_list = ['UCAGG_CCUAAG']


for i in double_mutants:
    idx = [0,0]
    counter = -1
    sequence = i
    mutation = compare_strings(i,wt_sequence)
    for j in range(len(mutation)):
        if mutation[j] == 0:
            counter += 1
            #choose the residues that need to be mutated
            wt_list = list(wt_sequence)
            if wt_list[j] == 'A':
                nucleotides = for_A
            elif wt_list[j] == 'C':
                nucleotides = for_C
            elif wt_list[j] == 'G':
                nucleotides = for_G
            elif wt_list[j] == 'U':
                nucleotides = for_U

            string = list(sequence)
            residue = string[j]
            
            if residue == nucleotides[0]:
                idx[counter] = (j) * 3 + 0
            if residue == nucleotides[1]:
                idx[counter] = (j) * 3 + 1
            if residue == nucleotides[2]:
                idx[counter] = (j) * 3 + 2
#            if residue == 'U':
#                idx_X = (j) * 4 + 3
    matrix[idx[0],idx[1]] = sequence
    matrix[idx[1],idx[0]] = sequence    

#%%
dM_sequence_matrix = pd.DataFrame(matrix)
dM_sequence_matrix.to_pickle('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/'
                             'submodule_coupling_11ntR/dM_sequence_matrix.pkl')

