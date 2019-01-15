#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:39:52 2019

@author: Steve
"""

'''Import libraries'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import pearsonr
from scipy.stats import norm
import math
import random
#%%
#import data
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
data = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
data = data.drop_duplicates(subset='seq')
#%%
#import natural receptor data with experimental R values
new_data_path = '/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/January_2019/'
summary_11ntR_filt_30Mg = pd.read_pickle(new_data_path + 'summary_11ntR_filt_30Mg.pkl')
summary_IC3_filt = pd.read_pickle(new_data_path + 'summary_IC3_filt.pkl')
summary_VC2_filt = pd.read_pickle(new_data_path + 'summary_VC2_filt.pkl')
#%%


dG_threshold = -6.5
bin_size = 0.25
limit_l = -12.5
limit_r = -6.5
bins = np.linspace(limit_l,limit_r,int((limit_r - limit_l)/bin_size) + 1)
#%%
error_threshold = 2

dG_30Mg = pd.concat([data['dG_Mut2_GAAA'].copy(),data['dGerr_Mut2_GAAA'].copy()],axis = 1)
print('original lenght: ' + str(len(dG_30Mg)))
dG_30Mg = dG_30Mg[dG_30Mg['dG_Mut2_GAAA'] < dG_threshold]
print('thresholded data lenght: ' + str(len(dG_30Mg)))

error_threshold = 1
dG_30Mg = dG_30Mg[dG_30Mg['dGerr_Mut2_GAAA'] < error_threshold]
print('thresholded data lenght: ' + str(len(dG_30Mg)))

figure_size = (6,6)
x_subplots = 5
y_subplots = 5
fig, axs = plt.subplots(x_subplots, y_subplots, sharey=True, sharex = True, 
                        tight_layout=True,figsize = figure_size)
axs = axs.ravel()
counter = -1

means = []
stds = []
dG_left = []
dG_right = []
for each_bin in bins[0:-1]:
    counter += 1
    print(str(each_bin) + ' ' + str(each_bin + bin_size))
    dG_left.append(each_bin)
    dG_right.append(each_bin + bin_size)
    mask = (dG_30Mg['dG_Mut2_GAAA'] >= each_bin) & (dG_30Mg['dG_Mut2_GAAA'] < each_bin + bin_size)
    next_data = dG_30Mg[mask]
    mean,std=norm.fit(next_data['dGerr_Mut2_GAAA'])
    means.append(mean)
    stds.append(std)
    axs[counter].hist(next_data['dGerr_Mut2_GAAA'],bins=30, normed=True)
    xmin, xmax = plt.xlim()
    xmin = 0
    xmax = 1.5
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, std)
    axs[counter].plot(x, y)
    txt_str = 'dG range ' + str(each_bin) + ' ' + str(each_bin + bin_size)
    a = round(mean,2)
    b = round(std,2)
    axs[counter].text(0.3,5,txt_str)
    axs[counter].text(0.3,4,'error mean: ' + str(a))
    axs[counter].text(0.3,3,'error std: ' + str(b))

error_dGs_norm = pd.DataFrame()
error_dGs_norm['dG_low_lim'] = dG_left
error_dGs_norm['dG_high_lim'] = dG_right
error_dGs_norm ['mean_error_dG'] = means
error_dGs_norm ['mean_error_std'] = stds 
#%% GET WT DATA TO BE USED AS REFERENCE 
receptor_data = data.groupby('r_seq')    
wt_data = receptor_data.get_group('UAUGG_CCUAAG')
wt_data = wt_data[wt_data['sublibrary'] != 'tertcontacts_5']
wt_data = wt_data[wt_data['b_name'] == 'normal']
wt_data = wt_data.set_index('old_idx')  
wt_data = wt_data['dG_Mut2_GAAA'].dropna()  
#%% Simulate data of specified ddG with specified error 
ddG_real = 3
mut_data = wt_data + ddG_real
mut_data_error = []
s_list = [] # errors added to theoretical mutant 
for values in mut_data:
    bin_number = (math.ceil((values - limit_l)/bin_size)) - 1 #determine bin size in which dG belongs
    if bin_number > (len (error_dGs_norm) - 1):
        mut_data_error.append(np.nan)
        s_list.append(np.nan)
    else:
        mu = error_dGs_norm.loc[bin_number]['mean_error_dG'] # get the mean error according to normal distribution above
        sigma = error_dGs_norm.loc[bin_number]['mean_error_std'] #get standard deviation
        s = float(np.random.normal(mu, sigma, 1))
        s_list.append(s)
        r = random.randint(0,1)
        if r == 0:
            mut_data_error.append(values + s)
        else:
            mut_data_error.append(values - s)

simulated_data = pd.DataFrame(index = wt_data.index)
simulated_data['wt_data'] = wt_data
simulated_data['theoretical_mut_data'] = mut_data
simulated_data['error added'] = s_list
simulated_data['mut_data_with_error'] = mut_data_error

x_data = simulated_data['wt_data'].copy()
x_data[x_data> -7.1] = np.nan

y_data = simulated_data['mut_data_with_error'].copy()
y_data[y_data> -7.1] = np.nan

r = x_data.corr(y_data)
print(r)

#%%
# Calculate R for a series of ddG values (ranging from 0 to 4 kcal/mol)
# plot R as a function of ddG 
ddG_list = np.linspace(0,4,21)
r_list = []
r_sq_list = []
for ddG_value in ddG_list:
    ddG_real = ddG_value 
    mut_data = wt_data + ddG_real
    mut_data_error = []
    s_list = [] # errors added to theoretical mutant 
    for values in mut_data:
        bin_number = (math.ceil((values - limit_l)/bin_size)) - 1 #determine bin size in which dG belongs
        if bin_number > (len (error_dGs_norm) - 1):
            mut_data_error.append(np.nan)
            s_list.append(np.nan)
        else:
            mu = error_dGs_norm.loc[bin_number]['mean_error_dG'] # get the mean error according to normal distribution above
            sigma = error_dGs_norm.loc[bin_number]['mean_error_std'] #get standard deviation
            s = float(np.random.normal(mu, sigma, 1))
            s_list.append(s)
            r = random.randint(0,1)
            if r == 0:
                mut_data_error.append(values + s)
            else:
                mut_data_error.append(values - s)
    
    simulated_data = pd.DataFrame(index = wt_data.index)
    simulated_data['wt_data'] = wt_data
    simulated_data['theoretical_mut_data'] = mut_data
    simulated_data['error added'] = s_list
    simulated_data['mut_data_with_error'] = mut_data_error
    
    x_data = simulated_data['wt_data'].copy()
    x_data[x_data> -7.1] = np.nan
    print(len(x_data.dropna()))
    
    y_data = simulated_data['mut_data_with_error'].copy()
    y_data[y_data> -7.1] = np.nan
    print(len(y_data.dropna()))
    print('-------')
    
    r = x_data.corr(y_data)
    r_list.append(r)
    r_sq_list.append(r**2)
plt.plot(ddG_list,r_list, '--', color = 'black')
plt.xlim([-0.5,4.5])
plt.ylim([0,1.2])
#%%

#how many samples should I take
n_samples = 100
ddG_list = np.linspace(0,4,21)
samples = np.zeros((len(ddG_list),n_samples))

for i in range(n_samples):
    r_list = []
    r_sq_list = []
    for ddG_value in ddG_list:
        ddG_real = ddG_value 
        mut_data = wt_data + ddG_real
        mut_data_error = []
        s_list = [] # errors added to theoretical mutant 
        for values in mut_data:
            bin_number = (math.ceil((values - limit_l)/bin_size)) - 1 #determine bin size in which dG belongs
            if bin_number > (len (error_dGs_norm) - 1):
                mut_data_error.append(np.nan)
                s_list.append(np.nan)
            else:
                mu = error_dGs_norm.loc[bin_number]['mean_error_dG'] # get the mean error according to normal distribution above
                sigma = error_dGs_norm.loc[bin_number]['mean_error_std'] #get standard deviation
                s = float(np.random.normal(mu, sigma, 1))
                s_list.append(s)
                r = random.randint(0,1)
                if r == 0:
                    mut_data_error.append(values + s)
                else:
                    mut_data_error.append(values - s)
        
        x_data = wt_data.copy()
        x_data[x_data> -7.1] = np.nan
        
        y_data = pd.Series(mut_data_error,index = x_data.index)
        y_data[y_data> -7.1] = np.nan
    
        r = x_data.corr(y_data)
        r_list.append(r)
        r_sq_list.append(r**2)
    samples[:,i] = r_sq_list  

simul_rsq_avg = samples.mean(axis = 1)

plt.plot(ddG_list,simul_rsq_avg, '--', color = 'black')
plt.scatter(summary_11ntR_filt_30Mg['ddG_30Mg'],summary_11ntR_filt_30Mg['r']**2,
            s = 120, edgecolor = 'black', color = 'blue')

plt.scatter(summary_IC3_filt['ddG_30Mg'],summary_IC3_filt['r']**2,
            s = 120, edgecolor = 'black', color = 'red')

plt.scatter(summary_VC2_filt['ddG_30Mg'],summary_VC2_filt['r']**2,
            s = 120, edgecolor = 'black', color = 'cyan')
plt.xlim([-0.5,4.5])
plt.ylim([0,1.2])





    
    
    


