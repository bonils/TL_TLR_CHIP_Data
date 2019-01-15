#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:16:24 2019

@author: Steve
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:47:46 2018

@author: Steve
"""
#%%
#%
'''Import libraries'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import pearsonr
#%%
VC2 = 'GUAGG_CCUAAAC'
IC3 = 'GAGGG_CCCUAAC'
TLR11ntR_AC = 'UAUGG_CCUACG'
U9G = 'UAGGG_CCUAAG'
bp_11ntR = 'UUAGG_CCUAAG'

#%%
#%
'''Import Data'''
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
figures_names = 'natural_IC3_variants_'

#% Plot without error bars

list_to_plot = ['GAGGG_CCCUAAC',
 'GAGGA_UCCUAAC',
 'GAGGG_CCCUGAC',
 'GAGGA_UUCUAAC',
 'GAGAA_UUCUAAC',
 'GAAGG_CCCUAAC']

ref_sequence = 'UAUGG_CCUAAG'
cond_to_plot = 'dG_Mut2_GAAA'

#colors for different chip piece lengths
color_8 = 'magenta'
color_9 = 'black'
color_10 = 'white'
color_11 = 'green'

dG_threshold = -7.1
low_lim = -14
high_lim = -5
figure_size = (6.5,6.5)

x = [low_lim, dG_threshold] #for plotting  y= x line
y_thres = [dG_threshold,dG_threshold]


#load data
data = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
data = data.drop_duplicates(subset='seq')
data = data[data['b_name'] == 'normal']
print('data length: ' +  str(len(data)))
data = data[data['sublibrary'] != 'tertcontacts_5']
print('sublibraries 1 to 4: ' + str(len(data)))
receptor_data= data.groupby('r_seq')

#all possible scaffolds

scaffolds = list(set(data['old_idx']))

#reference sequence 
ref_data = receptor_data.get_group(ref_sequence)
ref_data = ref_data.set_index('old_idx')
ref_data = ref_data.reindex(scaffolds)


Colors = ref_data.length.copy()
Colors[Colors == 8] = color_8
Colors[Colors == 9] = color_9
Colors[Colors == 10] = color_10
Colors[Colors == 11] = color_11



ddG_median_list = []
ddG_std_list = []
n_compared_list = []
x_subplots = 3
y_subplots = 3
fig, axs = plt.subplots(x_subplots, y_subplots, sharey=True, sharex = True, tight_layout=True,figsize = figure_size)
x_label = '$\Delta$G$^{ref}_{bind}$ (kcal/mol)'
y_label = '$\Delta$G$^{variant}_{bind}$ (kcal/mol)'
fig.text(0.5, 0.01, x_label, ha='center')
fig.text(0.01, 0.5, y_label, va='center', rotation='vertical')
axs = axs.ravel()
counter = -1
figure_counter = 0
for receptors in list_to_plot:
    counter += 1
    if counter < (x_subplots * y_subplots):
        next_data = receptor_data.get_group(receptors)
        next_data = next_data.set_index('old_idx')
        next_data = next_data.reindex(ref_data.index)

        axs[counter].scatter(ref_data[cond_to_plot],next_data[cond_to_plot],s=50,edgecolors='k',c=list(Colors.values))
        axs[counter].plot(x,x,':k')
        axs[counter].plot(x,y_thres,':k',linewidth = 0.5)
        axs[counter].plot(y_thres,x,':k',linewidth = 0.5)
        axs[counter].set_xlim(low_lim,high_lim)
        axs[counter].set_ylim(low_lim,high_lim)
        axs[counter].set_xticks(list(range(-14,-4,4)))
        axs[counter].set_yticks(list(range(-14,-4,4)))
        axs[counter].set_title(receptors,fontsize=8)

        x_data = ref_data[cond_to_plot].copy()
        y_data = next_data[cond_to_plot].copy()
        ddG = y_data.subtract(x_data)
        ddG_avg = ddG.median()
        data_points_total = min(len(x_data.dropna()),len(y_data.dropna(())))
        n_above_limit = len(pd.concat([x_data,y_data],axis = 1).dropna())
               
        r_pearson = x_data.corr(y_data)
        r_sq = r_pearson**2
        textstr = 'n=%.0f\nn$_{thr}$ =%.0f\n$\Delta\Delta G=%.2f$\n$\mathrm{R^2}=%.2f$' % (data_points_total,n_above_limit,ddG_avg, r_sq)
        textstr = '$\Delta\Delta G=%.2f$\n$\mathrm{R^2}=%.2f$' % (ddG_avg, r_sq)
        axs[counter].text(-9, -10, textstr, fontsize=7,
        verticalalignment='top')
        x_ddG = [low_lim + ddG_avg, dG_threshold]
        x_new = [low_lim, dG_threshold - ddG_avg]
        axs[counter].plot(x_new,x_ddG,'--r',linewidth = 3)

        
    else:
        figure_counter += 1
        fig.savefig('/Users/Steve/Desktop/Tecto_temp_figures_2/' + figures_names + str(figure_counter) + '.pdf')
        fig.text(0.5, 0.01, x_label, ha='center')
        fig.text(0.01, 0.5, y_label, va='center', rotation='vertical')
        fig, axs = plt.subplots(x_subplots, y_subplots, sharey=True, sharex = True, tight_layout=True, figsize = figure_size)
        fig.text(0.5, 0.04, x_label, ha='center')
        fig.text(0.04, 0.5, y_label, va='center', rotation='vertical')
        axs = axs.ravel()
                #plot 30 mM GAAA
        axs[0].scatter(ref_data[cond_to_plot],next_data[cond_to_plot],s=50,edgecolors='k',c=list(Colors.values))

        axs[0].plot(x,x,':k')
        axs[0].plot(x,y_thres,':k',linewidth = 0.5)
        axs[0].plot(y_thres,x,':k',linewidth = 0.5)
        axs[0].set_xlim(low_lim,high_lim)
        axs[0].set_ylim(low_lim,high_lim)
        axs[0].set_xticks(list(range(-14,-4,4)))
        axs[0].set_yticks(list(range(-14,-4,4)))
        axs[0].set_title(receptors,fontsize=8)

        x_data = ref_data[cond_to_plot].copy()
        y_data = next_data[cond_to_plot].copy()
        ddG = y_data.subtract(x_data)
        ddG_avg = ddG.median()
        data_points_total = min(len(x_data.dropna()),len(y_data.dropna(())))
        n_above_limit = len(pd.concat([x_data,y_data],axis = 1).dropna())
               
        r_pearson = x_data.corr(y_data)
        r_sq = r_pearson**2
        textstr = 'n=%.0f\nn$_{thr}$ =%.0f\n$\Delta\Delta G=%.2f$\n$\mathrm{R^2}=%.2f$' % (data_points_total,n_above_limit,ddG_avg, r_sq)
        textstr = '$\Delta\Delta G=%.2f$\n$\mathrm{R^2}=%.2f$' % (ddG_avg, r_sq)
        axs[0].text(-9, -10, textstr, fontsize=7,
        verticalalignment='top')
        x_ddG = [low_lim + ddG_avg, dG_threshold]
        x_new = [low_lim, dG_threshold - ddG_avg]              
        axs[0].plot(x_new,x_ddG,'--r',linewidth = 3)
        counter = 0
figure_counter += 1 
fig.savefig('/Users/Steve/Desktop/Tecto_temp_figures_2/' + figures_names + str(figure_counter) + '.pdf')




