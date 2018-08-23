#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:09:27 2018

@author: Steve
"""

'''--------------Import Libraries--------------------'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import *
import math
'''---------------General Variables-------------'''
dG_threshold = -7.1 #kcal/mol; dG values above this are not reliable
dG_replace = -7.1 # for replacing values above threshold. 
nan_threshold = 0.50 #amount of missing data tolerated.
num_neighbors = 10
#for plotting
low_lim = -14
high_lim = -5
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]
#%%
'''---------------Import data ------------------'''
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
all_data = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
all_data = all_data.drop_duplicates(subset='seq')

# Create uniquee name for each receptor. 
all_data['new_name'] = all_data['r_name'] + '_' + all_data['r_seq'] + '_' +  all_data['sublibrary']
all_data = all_data[all_data.b_name == 'normal']
all_data = all_data.set_index('new_name')
print(len(all_data))
#%%
#Import list of recepotors that are uncatalogued and that were removed from analysis
#and remove them from this analysis
uncatalogued_receptors = pd.read_csv('uncatalogued_receptors.csv')
uncatalogued_receptors = uncatalogued_receptors.set_index('new_name')
all_data = all_data.drop(uncatalogued_receptors.index)
print(len(all_data))
#%% A threshold for error on dG may be applied; if applied, these dGs and their
#errors are replaced by NaA

error_threshold = 1.25
#################
apply_error_threshold = False
#################

if apply_error_threshold:
    mask = all_data['dGerr_Mut2_GAAA'] > error_threshold
    all_data['dG_Mut2_GAAA'][mask] = np.nan
    all_data['dGerr_Mut2_GAAA'][mask] = np.nan
    print(sum(mask))
    
    mask = all_data['dGerr_Mut2_GAAA_5mM_2'] > error_threshold
    all_data['dG_Mut2_GAAA_5mM_2'][mask] = np.nan
    all_data['dGerr_Mut2_GAAA_5mM_2'][mask] = np.nan
    print(sum(mask))
    
    mask = all_data['dGerr_Mut2_GAAA_5mM_150mMK_1'] > error_threshold
    all_data['dG_Mut2_GAAA_5mM_150mMK_1'][mask] = np.nan
    all_data['dGerr_Mut2_GAAA_5mM_150mMK_1'][mask] = np.nan
    print(sum(mask))
    
    mask = all_data['dGerr_Mut2_GUAA_1'] > error_threshold
    all_data['dG_Mut2_GUAA_1'][mask] = np.nan
    all_data['dGerr_Mut2_GUAA_1'][mask] = np.nan
    print(sum(mask))

#Plot distribution of errors across entire library 
plt.figure()
plt.hist(all_data.dGerr_Mut2_GAAA.dropna())

plt.figure()
plt.hist(all_data.dGerr_Mut2_GAAA_5mM_2.dropna())

plt.figure()
plt.hist(all_data.dGerr_Mut2_GAAA_5mM_150mMK_1.dropna())

plt.figure()
plt.hist(all_data.dGerr_Mut2_GUAA_1.dropna())
#%% Group receptors by their name 
receptor_group = all_data.groupby(by='new_name')
#%% Create list with all scaffolds 
scaffolds = list(set(all_data['old_idx']))
#%% I wrote this to find out which are the most stable receptors in the library
#that were characterized with at leas 40 scaffolds.
dG_avg_30mM_GAAA = receptor_group['dG_Mut2_GAAA'].agg(['mean','count'])
dG_avg_30mM_GAAA_50scaff = dG_avg_30mM_GAAA[dG_avg_30mM_GAAA['count'] > 40]

#HOW MANY OF THE MOST STABLE
#########
n_stable = 200
#########
most_stable = dG_avg_30mM_GAAA_50scaff.nsmallest(n_stable,'mean')
#%% Create dataframe with WT data for comparisons below
WT_data = receptor_group.get_group('11ntR_UAUGG_CCUAAG_tertcontacts_0')
WT_data = pd.DataFrame(WT_data)
WT_data = WT_data.set_index('old_idx')
WT_data = WT_data.reindex(scaffolds)
#%% Compare WT to one of the receptors and plot

#to compare to one the most stable 
#########
which_mut  = 2
#########

#mutant has the index of the 'which mutant' most stable
mutant = most_stable.index[which_mut]
#or type the index of the desired receptors 
mutant = '11ntR_6A_7C_CCUGG_CCUAAA_tertcontacts_2'

#which datasets to plot
##################
x_to_plot = 'dG_Mut2_GAAA'
xerr_to_plot = 'dGerr_Mut2_GAAA'
y_to_plot = 'dG_Mut2_GAAA'
yerr_to_plot = 'dGerr_Mut2_GAAA'
##################

#Get data for desired mutant
mut_data = receptor_group.get_group(mutant)
mut_data = pd.DataFrame(mut_data)
mut_data = mut_data.set_index('old_idx')
mut_data = mut_data.reindex(scaffolds)

#get data to plot
x_data = WT_data[x_to_plot].copy()
x_err = WT_data[xerr_to_plot].copy()
y_data = mut_data[y_to_plot].copy()
y_err = mut_data[yerr_to_plot].copy()

#plot 
plt.errorbar(x_data, y_data, xerr = x_err,yerr = y_err,fmt='o' )
#%%Calculate pearson correlation and 95% confidence interval for mutant plotted
#THIS DOES NOT TAKE INTO ACCOUNT ERROR IN EACH MEASUREMENT

# And also by bootstrap method


#Calculate without considering values above dG_threshold?
###############
apply_dG_thr = True
dG_thr = dG_threshold
##############

#number of bootsrap interations
#############
n_bootstraps = 1000
#############


if apply_dG_thr:
    x_data[x_data > dG_thr] = np.nan
    y_data[y_data > dG_thr] = np.nan

r = x_data.corr(y_data)

n_total_x = len(x_data.dropna())
n_total_y = len(y_data.dropna())
n_compare = sum(~(x_data.isna() | y_data.isna()))
#calculate confidence interval without taking into account error in each measurement

#in normal space
z = math.log((1 + r)/(1-r)) * 0.5

#Standard error
SE = 1/math.sqrt(n_compare - 3)

#interval in normal space
low_z = z - (1.96 * SE)
high_z = z + (1.96 * SE)

#95% confidence interval
low_r = (math.exp(2 * low_z) - 1)/(math.exp(2 * low_z) + 1)
high_r = (math.exp(2 * high_z) - 1)/(math.exp(2 * high_z) + 1)

print ('By traditional method the pearson coeff. is :' + str(r))
print('and the 95% confidence interval is :' +  str(low_r) +  ' to ' + str(high_r))


#parametric bootstrap
combined_data = pd.DataFrame(index= x_data.index)
combined_data['x'] = x_data
combined_data['y'] = y_data
combined_data['x_err'] = x_err
combined_data['y_err'] = y_err
#eliminate NaNs because I am using numpy array functins that don't like NaNs

mask = ~(combined_data['x'].isna() | combined_data['y'].isna())
combined_data = combined_data[mask]
n_comb = len(combined_data)

list_corr = []
for i in range(n_bootstraps):
    x = np.random.normal(combined_data['x'],combined_data['x_err'])
    y = np.random.normal(combined_data['y'],combined_data['y_err'])
    r_bs = np.corrcoef(x,y)[0,1]   
    list_corr.append(r_bs)
    
list_corr = np.array(list_corr)
list_corr.sort()
mean_r_bs = list_corr.mean()
median_r_bs = np.median(list_corr)
low_r_bs = list_corr[int(0.05 * n_bootstraps) -1]
high_r_bs = list_corr[-1*(int(0.05 * n_bootstraps) -1)]

print ('By bootstrap method the pearson coeff. is :' + str(mean_r_bs))
print('and the 95% confidence interval is :' +  str(low_r_bs) +  ' to ' + str(high_r_bs))
#%% Write a function that calculates bootstraps pearson correlation coefficients
#and 95% confidence interval 
def pearson_bootsrap (x_data,y_data,n_boostraps):
    #x_data is dataframe indexed by 'old_idx' usually containing WT data
    #y_data is dataframe equally indexed, usually containing mut data
    #x_data and y_data containg 2 columns first one has dGs and the second one
    #has the errors. 
    #n_bootsraps is number of iterations for bootstras
    combined_data = pd.DataFrame(index= x_data.index)
    combined_data['x'] = x_data[x_data.columns[0]]
    combined_data['y'] = y_data[y_data.columns[0]]
    combined_data['x_err'] = x_data[x_data.columns[1]]
    combined_data['y_err'] = y_data[y_data.columns[1]]
    mask = ~(combined_data['x'].isna() | combined_data['y'].isna())
    combined_data = combined_data[mask]
    n_comb = len(combined_data)
    list_corr = []
    for i in range(n_bootstraps):
        x = np.random.normal(combined_data['x'],combined_data['x_err'])
        y = np.random.normal(combined_data['y'],combined_data['y_err'])
        r_bs = np.corrcoef(x,y)[0,1]   
        list_corr.append(r_bs)
    list_corr = np.array(list_corr)
    list_corr.sort()
    mean_r_bs = list_corr.mean()
    median_r_bs = np.median(list_corr)
    low_r_bs = list_corr[int(0.05 * n_bootstraps) -1]
    high_r_bs = list_corr[-1*(int(0.05 * n_bootstraps) -1)]
    return (mean_r_bs,median_r_bs,low_r_bs,high_r_bs,n_comb)


#%%Function to calculate Pearson correlations with confidence interval without
#taking into account errors     
def Pearson_and_confidence(x_data,y_data):
    
    r = x_data[x_data.columns[0]].corr(y_data[y_data.columns[0]])
    n_compare = sum(~(x_data[x_data.columns[0]].isna() | y_data[y_data.columns[0]].isna()))
    #calculate confidence interval without taking into account error in each measurement
    
    #in normal space
    z = math.log((1 + r)/(1-r)) * 0.5
    
    #Standard error
    SE = 1/math.sqrt(n_compare - 3)
    
    #interval in normal space
    low_z = z - (1.96 * SE)
    high_z = z + (1.96 * SE)
    
    #95% confidence interval
    low_r = (math.exp(2 * low_z) - 1)/(math.exp(2 * low_z) + 1)
    high_r = (math.exp(2 * high_z) - 1)/(math.exp(2 * high_z) + 1)
    return (r,low_r,high_r,n_compare)
#%%Calculate pearson coeffs for all of the receptors that have more than n=40
#relative to WT    
R_pearson_stats = pd.DataFrame(index = dG_avg_30mM_GAAA_50scaff.index)

mean_r_bs_list = []
median_r_bs_list = []
low_r_bs_list = []
high_r_bs_list = []
n_list = []
mean_dG_list = []

r_list = []
low_r_list = []
high_r_list = []
n_compare_list = []

n_bootstraps = 1000

WT_data = receptor_group.get_group('11ntR_UAUGG_CCUAAG_tertcontacts_0')
WT_data = pd.DataFrame(WT_data)
WT_data = WT_data.set_index('old_idx')
WT_data = WT_data.reindex(scaffolds)
x_to_plot = 'dG_Mut2_GAAA'
xerr_to_plot = 'dGerr_Mut2_GAAA'
y_to_plot = 'dG_Mut2_GAAA'
yerr_to_plot = 'dGerr_Mut2_GAAA'
x_data = WT_data[[x_to_plot,xerr_to_plot]].copy()

######
x_data[x_data[x_to_plot] >dG_thr] = np.nan
######


for receptors in dG_avg_30mM_GAAA_50scaff.index:
    #get data for mutant
    mut_data = receptor_group.get_group(receptors).copy()
    mut_data = pd.DataFrame(mut_data)
    mut_data = mut_data.set_index('old_idx')
    mut_data = mut_data.reindex(scaffolds)
    y_data = mut_data[[y_to_plot,yerr_to_plot]].copy()
    ##########
    y_data[y_data[y_to_plot]>dG_threshold] = np.nan
    ##########
    
    mean_dG_list.append(y_data[y_to_plot].mean())
    ##########
    mean_r_bs,median_r_bs,low_r_bs,high_r_bs,n_comb = pearson_bootsrap(x_data,y_data,n_bootstraps)

    mean_r_bs_list.append(mean_r_bs)
    median_r_bs_list.append(median_r_bs)
    low_r_bs_list.append(low_r_bs)
    high_r_bs_list.append(high_r_bs)
    n_list.append(n_comb)
    
    #now using normal (without taking into account error)
    r,low_r,high_r,n_compare = Pearson_and_confidence(x_data,y_data)
    r_list.append(r)
    low_r_list.append(low_r)
    high_r_list.append(high_r)
    n_compare_list.append(n_compare)
    

R_pearson_stats['n_compared'] = n_list
R_pearson_stats['mean_dG'] = mean_dG_list
R_pearson_stats['bootstrap_r_mean'] = mean_r_bs_list
R_pearson_stats['bootstrap_r_median'] = median_r_bs_list
R_pearson_stats['bootstrap_r_low'] = low_r_bs_list
R_pearson_stats['bootstrap_r_high'] = high_r_bs_list

R_pearson_stats['r_pearson'] = r_list
R_pearson_stats['r_low'] = low_r_list
R_pearson_stats['r_high'] = high_r_list
R_pearson_stats['n_comp'] = n_compare_list
#%%
plt.scatter(R_pearson_stats['r_pearson'],R_pearson_stats['bootstrap_r_median'])
plt.plot(R_pearson_stats['r_pearson'],R_pearson_stats['r_pearson'])
plt.scatter(R_pearson_stats['r_pearson'],R_pearson_stats['bootstrap_r_mean'])
plt.ylim([0,1])
plt.xlim([0,1])
#%%
plt.scatter(R_pearson_stats['mean_dG'],R_pearson_stats['bootstrap_r_mean'])
plt.scatter(R_pearson_stats['mean_dG'],R_pearson_stats['r_pearson'])
plt.ylim([-1,1])
#%%
#more than 15 measurements
A = R_pearson_stats[R_pearson_stats.n_compared> 20]
plt.scatter(A['mean_dG'],A['bootstrap_r_mean'])
plt.ylim([0,1])
#%%
rand_idx = A.index[np.random.randint(len(A))]
wt_data = WT_data.reindex(scaffolds)
mut_data = receptor_group.get_group(rand_idx)
mut_data = mut_data.set_index('old_idx')
mut_data = mut_data.reindex(scaffolds)

plt.errorbar(wt_data['dG_Mut2_GAAA'],mut_data['dG_Mut2_GAAA'],
             xerr = wt_data['dGerr_Mut2_GAAA'],yerr = mut_data['dGerr_Mut2_GAAA'],
             fmt = 'o')

low_lim = -14
high_lim = -6
x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]

plt.plot(x,x,':k')
plt.plot(x,y_thres,':k',linewidth = 0.5)
plt.plot(y_thres,x,':k',linewidth = 0.5)
plt.xlim(low_lim,high_lim)
plt.ylim(low_lim,high_lim)
plt.xticks(list(range(-14,-4,2)))
plt.yticks(list(range(-14,-4,2)))
plt.tick_params(axis='both', which='major', labelsize=24)
plt.axes().set_aspect('equal')
plt.xlabel('$\Delta$G$^{11ntR}_{bind}$ (kcal/mol)',fontsize=22)
plt.ylabel('$\Delta$G$^{mut}_{bind}$ (kcal/mol)',fontsize=22)
plt.title(rand_idx)



 #%%
WT_data = receptor_group.get_group('11ntR_UAUGG_CCUAAG_tertcontacts_0')
WT_data = pd.DataFrame(WT_data)
WT_data = WT_data.set_index('old_idx')
x_to_plot = 'dG_Mut2_GAAA'
xerr_to_plot = 'dGerr_Mut2_GAAA'
y_to_plot = 'dG_Mut2_GAAA'
yerr_to_plot = 'dGerr_Mut2_GAAA'
x_data = WT_data[[x_to_plot,xerr_to_plot]]
 
mut_data = receptor_group.get_group(receptors)
mut_data = pd.DataFrame(mut_data)
mut_data = mut_data.set_index('old_idx')
mut_data = mut_data.reindex(scaffolds)
y_data = mut_data[[y_to_plot,yerr_to_plot]]

mean_r,low_r,high_r,n_comb = pearson_bootsrap(x_data,y_data,n_bootstraps)
 