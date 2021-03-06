#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:47:46 2018

@author: Steve
"""
VC2 = 'GUAGG_CCUAAAC'
IC3 = 'GAGGG_CCCUAAC'
TLR11ntR_AC = 'UAUGG_CCUACG'
U9G = 'UAGGG_CCUAAG'
bp_11ntR = 'UUAGG_CCUAAG'


'''Sequence to plot wrt to 11ntR'''
alt_TLR_seq = VC2
#'UAGGA_UCUAAG'
#maybe a favorite 'CAUGG_CCUACG'
#'UAGGG_CCUAAG'
#'CAUGG_CCUACG' --> triple mutants with ~ 2 kcal/mol effect 

color_to_plot = 'navy'


'''Condition to compare'''
condition = 'dG_Mut2_GAAA'  # for 30 mM Mg
error = 'dGerr_Mut2_GAAA'

condition2 = 'dG_Mut2_GAAA' 
error2 = 'dGerr_Mut2_GAAA'

#'dG_Mut2_GAAA' 
#'dG_Mut2_GAAA_5mM_150mMK_1'

#plot GUAA too??

plot_GUAA = False



'''Limits for plot figures'''
low_lim = -14
high_lim = -6

'''Set threshold'''
dG_threshold = -7.1
set_threshold = True

'''plot originial or thresholded data'''
plot_original = True

#'AA_AU',
# 'AC_GU',
# 'AG_CU',
# 'CC_GG',
# 'CG_CG',
# 'GC_GC',
# 'GG_CC',
# 'GU_AC',
# 'UC_GG'}



#%%
'''Import libraries'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import pearsonr
#%%
'''Import Data'''
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
data_11ntR = pd.read_csv(data_path + 'all_11ntRs_unique.csv')
print('original size of dataframe with all 11ntR is:',data_11ntR.shape)
#Select data only with normal flanking base pairs
data_11ntR = data_11ntR[data_11ntR.b_name == 'normal']
print('data with normal flaking base pairs is: ', data_11ntR.shape)
#%%
entire_lib = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')
entire_lib = entire_lib.drop_duplicates(subset='seq')
entire_lib = entire_lib[entire_lib['b_name'] == 'normal']
grouped_lib = entire_lib.groupby('r_seq')
#%%
GUAA_vs_GAAA = grouped_lib['dG_Mut2_GAAA','dG_Mut2_GUAA_1']
GUAA_vs_GAAA = GUAA_vs_GAAA.agg(['median','count'])
GUAA_vs_GAAA = GUAA_vs_GAAA[GUAA_vs_GAAA['dG_Mut2_GAAA']['count'] > 4]

#plt.scatter(range(len(GUAA_vs_GAAA)),GUAA_vs_GAAA['dG_Mut2_GAAA']['median'])
plt.scatter(GUAA_vs_GAAA['dG_Mut2_GAAA']['median'],GUAA_vs_GAAA['dG_Mut2_GUAA_1']['median'])

#%%
sublib2 = entire_lib[entire_lib.sublibrary == 'tertcontacts_2']
contexts = list(set(sublib2.old_idx))
sequences = list(set(sublib2.r_seq))
seq_right = [section[0:5]+ ';' for section in sequences]
seq_left = [section[6:]+ ';' for section in sequences]
#%%
'''WT'''
# create dataframe with WT receptors and flanking bp == 'normal'
WT_11ntR = data_11ntR[data_11ntR.r_seq == 'UAUGG_CCUAAG']
WT_11ntR = grouped_lib.get_group('UAUGG_CCUAAG')
print('size of WT dataframe:',WT_11ntR.shape)
#%%
unique_scaffolds = set(WT_11ntR['old_idx'])
WT_11ntR = WT_11ntR.set_index('old_idx')
WT_11ntR = WT_11ntR.reindex(unique_scaffolds)
#%%
alt_TLR = data_11ntR[data_11ntR.r_seq == alt_TLR_seq]
alt_TLR = grouped_lib.get_group(alt_TLR_seq)
alt_TLR = alt_TLR.set_index('old_idx')
alt_TLR = alt_TLR.reindex(unique_scaffolds)
#%% Keep original data without thresholding for plotting.
WT_11ntR_original = WT_11ntR.copy()
alt_TLR_original = alt_TLR.copy()
#%%
'''Set values above dG threshold to NAN'''
if set_threshold:
    cond = alt_TLR[condition2].copy()
    cond[cond>dG_threshold] = np.nan
    alt_TLR[condition2] = cond
    del cond
    cond = WT_11ntR[condition].copy()
    cond[cond>dG_threshold] = np.nan
    WT_11ntR[condition] = cond
#%%
#ddG and correlation values are calculated based on thresholfed data
'''Calculate ddG values'''
ddG = alt_TLR[condition2] - WT_11ntR[condition] 
ddG_average = ddG.mean()
print('ddG_avg using values < threshold: ' + str(ddG_average))
ddG_std = ddG.std()


#calculate ddG median taking into account all values
ddG_2 = alt_TLR_original[condition2] - WT_11ntR_original[condition]
ddG_median = ddG_2.median()
ddG_median_std = ddG_2.std()
print('ddG median using all values: ' + str(ddG_median))


'''Correlation coefficient of thresholded data'''
data_comb = pd.concat([alt_TLR[condition2],WT_11ntR[condition]],axis = 1)
R = data_comb.corr(method = 'pearson')
#calculate rmse with respect to the median
model = WT_11ntR[condition] + ddG_median
model_df = pd.concat([alt_TLR[condition2],model],axis = 1)
model_df = model_df.dropna()
model_df.columns = [['actual','model']]
rms = sqrt(mean_squared_error(model_df['actual'],model_df['model']))
'''Correlation coefficient original data'''
data_comb_original = pd.concat([alt_TLR_original[condition2],WT_11ntR_original[condition]],axis = 1)
data_comb_original.columns = ['column_1','column_2']
R_original = data_comb_original.corr(method = 'pearson')
data_comb_original[data_comb_original>-7.1] = np.nan
data_clean = data_comb_original.dropna()
r,p = pearsonr(data_clean['column_1'],data_clean['column_2'])
#%%
#calculate RMSE
data = alt_TLR[condition].dropna().copy()
diff = data - ddG_average

#%%
#Color based on the length of the CHIP piece 
Colors = WT_11ntR.length.copy()
Colors[Colors == 8] = 'magenta'#color_to_plot#'red'
Colors[Colors == 9] = 'black'#color_to_plot#'blue'
Colors[Colors == 10] = 'white' #color_to_plot#'orange'
Colors[Colors == 11] = 'green' #color_to_plot#'black'

#%%
if plot_original:
   fig1 = plt.figure()
   x = [low_lim, dG_threshold] #for plotting  y= x line
   y_thres = [dG_threshold,dG_threshold]
   x_ddG = [ddG_average + x[0],ddG_average + x[1]]
   plt.plot(x,x_ddG,'--r',linewidth = 3)
   plt.scatter(WT_11ntR_original[condition],alt_TLR_original[condition2],s=120,edgecolors='k',c=Colors,linewidth=2)
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
   fig1.show() 
else:
    fig1 = plt.figure()
    x = [low_lim, dG_threshold] #for plotting  y= x line
    y_thres = [dG_threshold,dG_threshold]
    x_ddG = [ddG_average + x[0],ddG_average + x[1]]
    plt.plot(x,x_ddG,'--r',linewidth = 3)
    plt.scatter(WT_11ntR[condition],alt_TLR[condition2],s=120,edgecolors='k',c=Colors)
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
    fig1.show()

fig1.savefig(alt_TLR_seq + '.svg')
print('ddG_average: ' + str(ddG_average))
print('ddG_std: ' + str(ddG_std))
#print('R_sq taking only values within limit: ' + str(R))
    
#%%
plt.figure(figsize=(5,5))
wt_data = grouped_lib.get_group('UAUGG_CCUAAG')
wt_data_l = wt_data.groupby('length')
lengths= [8,9,10,11]
medians = []
stds = []
for length in lengths:
    medians.append(wt_data_l.get_group(length)[condition].median())
    stds.append(wt_data_l.get_group(length)[condition].std())
plt.scatter(lengths,medians,marker='s', s = 160,edgecolors='k')
plt.errorbar(lengths,medians,yerr=stds)

variant_data = grouped_lib.get_group(alt_TLR_seq)
variant_data_l = variant_data.groupby('length')
lengths= [8,9,10,11]
medians = []
stds = []
for length in lengths:
    medians.append(variant_data_l.get_group(length)[condition2].median())
    stds.append(variant_data_l.get_group(length)[condition2].std())
plt.scatter(lengths,medians,marker='s',c=color_to_plot, s= 160,edgecolors='k',linewidth = 2)
plt.errorbar(lengths,medians,c='black',yerr=stds)
#plt.xlim(7.5,11.5)
plt.ylim(-14,-6)
#plt.axes().set_aspect('equal')
#plt.set_size_inches(5,5)

print(rms)

#%% get median dG for each of the receptors 
plt.figure()
all_scaffolds = list(set(wt_data['old_idx']))
TLR_11ntR_GAAA = grouped_lib.get_group('UAUGG_CCUAAG')['dG_Mut2_GAAA']
TLR_IC3_GAAA = grouped_lib.get_group(IC3)['dG_Mut2_GAAA']
TLR_VC2_GAAA = grouped_lib.get_group(VC2)['dG_Mut2_GAAA']
TLR_U9G_GAAA = grouped_lib.get_group(U9G)['dG_Mut2_GAAA']

TLR_11ntR_GUAA = grouped_lib.get_group('UAUGG_CCUAAG')['dG_Mut2_GUAA_1']
TLR_IC3_GUAA = grouped_lib.get_group(IC3)['dG_Mut2_GUAA_1']
TLR_VC2_GUAA = grouped_lib.get_group(VC2)['dG_Mut2_GUAA_1']
TLR_U9G_GUAA = grouped_lib.get_group(U9G)['dG_Mut2_GUAA_1']
TLR_bp11nTR_GUAA = grouped_lib.get_group(bp_11ntR)['dG_Mut2_GUAA_1']


TLR_medians = [TLR_11ntR_GAAA.median(),TLR_11ntR_GUAA.median(),TLR_IC3_GAAA.median(),TLR_IC3_GUAA.median(),TLR_VC2_GAAA.median(),TLR_VC2_GUAA.median()]
TLR_std = [TLR_11ntR_GAAA.std(),TLR_11ntR_GUAA.std(),TLR_IC3_GAAA.std(),TLR_IC3_GUAA.std(),TLR_VC2_GAAA.std(),TLR_VC2_GUAA.std()]
plt.bar(range(6),TLR_medians,yerr=TLR_std)
plt.ylim(-14,-6)




plt.figure()
TLR_medians = [TLR_bp11nTR_GUAA.median(),TLR_11ntR_GAAA.median(),TLR_IC3_GAAA.median(),TLR_VC2_GAAA.median(),TLR_U9G_GAAA.median()]
TLR_std = [TLR_bp11nTR_GUAA.std(),TLR_11ntR_GAAA.std(),TLR_IC3_GAAA.std(),TLR_VC2_GAAA.std(),TLR_U9G_GAAA.std()]
plt.bar(range(5),TLR_medians,yerr=TLR_std)
plt.ylim(-14,-6)


#%% plot ddG instead 

wt_data = grouped_lib.get_group('UAUGG_CCUAAG')
wt_data = wt_data.set_index('old_idx')
wt_data = wt_data.reindex(all_scaffolds)
IC3_data = grouped_lib.get_group(IC3)
IC3_data = IC3_data.set_index('old_idx')
IC3_data = IC3_data.reindex(all_scaffolds)
VC2_data = grouped_lib.get_group(VC2)
VC2_data = VC2_data.set_index('old_idx')
VC2_data = VC2_data.reindex(all_scaffolds)


ddG_IC3 = IC3_data['dG_Mut2_GAAA'].subtract(wt_data['dG_Mut2_GAAA']).median()
std_IC3 = IC3_data['dG_Mut2_GAAA'].subtract(wt_data['dG_Mut2_GAAA']).std()

ddG_VC2 = VC2_data['dG_Mut2_GAAA'].subtract(wt_data['dG_Mut2_GAAA']).median()
std_VC2 = VC2_data['dG_Mut2_GAAA'].subtract(wt_data['dG_Mut2_GAAA']).std()

plt.figure()
ddGs = [ddG_IC3,ddG_VC2]
stds = [std_IC3,std_VC2]
plt.bar(range(2),ddGs,yerr=stds)




