#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:45:24 2019

@author: Steve
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import pearsonr
import math
#%%
#--------------------------
#Variables
dG_thr = -7.1
#-------------------------

#%
#Functions     
def Pearson_and_confidence(x_data,y_data):
#Function to calculate Pearson correlations with confidence interval without
#taking into account errors    
    #r = x_data[x_data.columns[0]].corr(y_data[y_data.columns[0]])
    #n_compare = sum(~(x_data[x_data.columns[0]].isna() | y_data[y_data.columns[0]].isna()))
    #calculate confidence interval without taking into account error in each measurement
    
    r = x_data.corr(y_data)
    n_compare = len(x_data)
    if n_compare > 3:
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
    else:
        low_r = np.nan
        high_r = np.nan
        
    return (r,low_r,high_r,n_compare)


def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)
#%
#import binding data
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
entire_lib = pd.read_csv(data_path + 'tectorna_results_tertcontacts.180122.csv')

entire_lib = entire_lib.drop_duplicates(subset ='seq')
#Consider the ones only with normal closing base pair 
mask = entire_lib.b_name == 'normal' 
entire_lib_normal_bp = entire_lib[mask]
entire_lib_selected = entire_lib_normal_bp[entire_lib_normal_bp['sublibrary'] != 'tertcontacts_5']
#%
#import all_receptors being analyzed 
receptors_types_matrix = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/receptors_types_matrix.pkl')
receptors_info = pd.read_pickle('/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/receptor_info.pkl')
#%%
# Created a new folder for new analyis 
new_data_path = '/Users/Steve/Desktop/Data_analysis_code/TL_TLR_CHIP_Data/January_2019/'
details_11ntR = pd.read_csv(new_data_path + 'details_11ntR_updated_01_09_2019.csv' )
details_11ntR['receptor_sequence'] = details_11ntR['receptor_sequence'].str.strip("'")
details_11ntR['receptor'] = details_11ntR['receptor_sequence']
#select only the ones that were found to have a GNRA receptor 
details_11ntR = details_11ntR[details_11ntR['GNRA_found'] == 1]
print ('original total number of 11ntR variants instances found: ' + str(len(details_11ntR)))
#select only the ones that have their secondary structure confirmed 
details_11ntR = details_11ntR[details_11ntR['confirmed'] == 1]
print('after visual confirmation of secondary structures: ' + str(len(details_11ntR)))
#%
#------ONLY NEED TO DO THIS ONCE, BEFORE UPDATING IT WITH VISUAL CONFIRMATION----------
#details_IC3 = pd.read_csv(data_path + 'details_IC3_found.csv' )
#details_IC3['receptor_sequence'] = details_IC3['receptor_sequence'].str.strip("'")
#details_IC3['receptor']  = details_IC3['receptor_sequence']
#details_IC3 = details_IC3[details_IC3['GNRA_found'] == 1]
#
##count totals for each sequence
#IC3_receptors = details_IC3.groupby('receptor')
#counts = []
#for sequence in details_IC3['receptor']:
#    counts.append(len(IC3_receptors.get_group(sequence)))
#details_IC3['total_found'] = counts
#details_IC3.to_csv(data_path + 'details_IC3.csv')
##
#details_VC2 = pd.read_csv(data_path + 'details_VC2_found.csv' )
#details_VC2['receptor_sequence'] = details_VC2['receptor_sequence'].str.strip("'")
#details_VC2['receptor']  = details_VC2['receptor_sequence']
#details_VC2 = details_VC2[details_VC2['GNRA_found'] == 1]
#
##count totals for each sequence
#VC2_receptors = details_VC2.groupby('receptor')
#counts = []
#for sequence in details_VC2['receptor']:
#    counts.append(len(VC2_receptors.get_group(sequence)))
#details_VC2['total_found'] = counts
#details_VC2.to_csv(data_path + 'details_VC2.csv')
#%
details_IC3 = pd.read_csv(new_data_path + 'details_IC3_updated_01_10_2019.csv' )
#select only the ones that were found to have a GNRA receptor 
details_IC3 = details_IC3[details_IC3['GNRA_found'] == 1]
print ('original total number of IC3 variants instances found: ' + str(len(details_IC3)))
#select only the ones that have their secondary structure confirmed 
details_IC3 = details_IC3[details_IC3['confirmed'] == 1]
print('after visual confirmation of secondary structures: ' + str(len(details_IC3)))
#%
details_VC2 = pd.read_csv(new_data_path + 'details_VC2_updated_01_10_2019.csv' )
#select only the ones that were found to have a GNRA receptor 
details_VC2 = details_VC2[details_VC2['GNRA_found'] == 1]
print ('original total number of VC2 variants instances found: ' + str(len(details_VC2)))
#select only the ones that have their secondary structure confirmed 
details_VC2 = details_VC2[details_VC2['confirmed'] == 1]
print('after visual confirmation of secondary structures: ' + str(len(details_VC2)))
#
#%% Get summary of ddGs, and frequencies for natural 11ntRs


#--------------------------11ntR----------------------------------------
data = entire_lib_selected.groupby('r_seq')
wt_data = data.get_group('UAUGG_CCUAAG')
wt_data = wt_data.set_index('old_idx')
scaffolds = wt_data.index
list_11ntR = list(set(details_11ntR['receptor_sequence']))
summary_11ntR = pd.DataFrame(index = list_11ntR)

ddG_30Mg = []
ddG_5Mg = []
ddG_5Mg150K = [] 

std_30Mg = []
std_5Mg = []
std_5Mg150K = []

R_30Mg = []
R_5Mg = []
R_5Mg150K = []


n_30Mg = []
n_5Mg = []
n_5Mg150K = []

n_R_30Mg = []
n_R_5Mg = []
n_R_5Mg150K = []

r = []
r_5Mg = []
r_5Mg150K = []


r_low = []
r_low_5Mg = []
r_low_5Mg150K = []


r_high = []
r_high_5Mg = []
r_high_5Mg150K = []


n_compared = []
n_compared_5Mg = []
n_compared_5Mg150K = []

for sequences in summary_11ntR.index:
    next_data = data.get_group(sequences)
    next_data = next_data.set_index('old_idx')
    next_data = next_data.reindex(scaffolds)
    
    #calculate ddGs, standard deviations, and number of variants in library
    ddG = next_data['dG_Mut2_GAAA'].subtract(wt_data['dG_Mut2_GAAA'])
    ddG_30Mg.append(ddG.median())
    std_30Mg.append(ddG.std())
    n_30Mg.append(len(next_data['dG_Mut2_GAAA'].dropna()))
    

    ddG = next_data['dG_Mut2_GAAA_5mM_2'].subtract(wt_data['dG_Mut2_GAAA_5mM_2'])
    ddG_5Mg.append(ddG.median())
    std_5Mg.append(ddG.std())
    n_5Mg.append(len(next_data['dG_Mut2_GAAA_5mM_2'].dropna()))    
    
    
    ddG = next_data['dG_Mut2_GAAA_5mM_150mMK_1'].subtract(wt_data['dG_Mut2_GAAA_5mM_150mMK_1'])
    ddG_5Mg150K.append(ddG.median())
    std_5Mg150K.append(ddG.std())
    n_5Mg150K.append(len(next_data['dG_Mut2_GAAA_5mM_150mMK_1'].dropna()))  
    
    
    #for calculating R coefficient only take values that are within threshold
    x = next_data['dG_Mut2_GAAA'].copy()
    x[x>dG_thr] = np.nan
    y = wt_data['dG_Mut2_GAAA'].copy()
    y[y>dG_thr] = np.nan
    R_data = pd.DataFrame()
    R_data['x'] = x
    R_data['y'] = y
    R_data = R_data.dropna()
    n_R_30Mg.append(len(R_data))
    corr = R_data.corr()
    R_30Mg.append(corr['x']['y']) 
    #Calculate errors for R values 
    x_data = R_data['x']
    y_data = R_data['y']
    r_,r_low_,r_high_,n_compared_ = Pearson_and_confidence(x_data,y_data)
    r.append(r_)
    r_low.append(r_low_)
    r_high.append(r_high_)
    n_compared.append(n_compared_)
    

    #repeat at 5 mM  Mg
    x = next_data['dG_Mut2_GAAA_5mM_2'].copy()
    x[x>dG_thr] = np.nan
    y = wt_data['dG_Mut2_GAAA'].copy()
    y[y>dG_thr] = np.nan
    R_data = pd.DataFrame()
    R_data['x'] = x
    R_data['y'] = y
    R_data = R_data.dropna()
    n_R_5Mg.append(len(R_data))
    corr = R_data.corr()
    R_5Mg.append(corr['x']['y']) 
    #Calculate errors for R values 
    x_data = R_data['x']
    y_data = R_data['y']
    r_,r_low_,r_high_,n_compared_ = Pearson_and_confidence(x_data,y_data)
    r_5Mg.append(r_)
    r_low_5Mg.append(r_low_)
    r_high_5Mg.append(r_high_)
    n_compared_5Mg.append(n_compared_)   
    
    
    #repeat at 5 mM  Mg + 150 K
    x = next_data['dG_Mut2_GAAA_5mM_150mMK_1'].copy()
    x[x>dG_thr] = np.nan
    y = wt_data['dG_Mut2_GAAA_5mM_150mMK_1'].copy()
    y[y>dG_thr] = np.nan
    R_data = pd.DataFrame()
    R_data['x'] = x
    R_data['y'] = y
    R_data = R_data.dropna()
    n_R_5Mg150K.append(len(R_data))
    corr = R_data.corr()
    R_5Mg150K.append(corr['x']['y']) 
    #Calculate errors for R values 
    x_data = R_data['x']
    y_data = R_data['y']
    r_,r_low_,r_high_,n_compared_ = Pearson_and_confidence(x_data,y_data)
    r_5Mg150K.append(r_)
    r_low_5Mg150K.append(r_low_)
    r_high_5Mg150K.append(r_high_)
    n_compared_5Mg150K.append(n_compared_)      
    
    
#fill up summary for 30 mM Mg     
summary_11ntR['ddG_30Mg'] = ddG_30Mg    
summary_11ntR['std_30Mg'] = std_30Mg  
summary_11ntR['n_30Mg'] = n_30Mg  
summary_11ntR['R_30Mg'] = R_30Mg  
summary_11ntR['n_R_30Mg'] = n_R_30Mg  
summary_11ntR['r'] = r 
summary_11ntR['r_low'] = r_low 
summary_11ntR['r_high'] = r_high
summary_11ntR['n_compared'] = n_compared

#fill up summary for 5 mM Mg 
summary_11ntR['ddG_5Mg'] = ddG_5Mg    
summary_11ntR['std_5Mg'] = std_5Mg  
summary_11ntR['n_5Mg'] = n_5Mg  
summary_11ntR['R_5Mg'] = R_5Mg  
summary_11ntR['n_R_5Mg'] = n_R_5Mg  
summary_11ntR['r_5Mg'] = r_5Mg
summary_11ntR['r_low_5Mg'] = r_low_5Mg 
summary_11ntR['r_high_5Mg'] = r_high_5Mg
summary_11ntR['n_compared_5Mg'] = n_compared_5Mg

#fill up summary for 5 mM Mg + 150 mM K
summary_11ntR['ddG_5Mg150K'] = ddG_5Mg150K    
summary_11ntR['std_5Mg150K'] = std_5Mg150K  
summary_11ntR['n_5Mg150K'] = n_5Mg150K  
summary_11ntR['R_5Mg150K'] = R_5Mg150K  
summary_11ntR['n_R_5Mg150K'] = n_R_5Mg150K  
summary_11ntR['r_5Mg150K'] = r_5Mg150K
summary_11ntR['r_low_5Mg150K'] = r_low_5Mg150K 
summary_11ntR['r_high_5Mg150K'] = r_high_5Mg150K
summary_11ntR['n_compared_5Mg150K'] = n_compared_5Mg150K

#fill up sequence frequency in natural RNAs
natural_11ntR = details_11ntR.groupby('receptor_sequence')
frequency = []
for sequence in summary_11ntR.index:
    frequency.append(len(natural_11ntR.get_group(sequence)))
summary_11ntR['frequency'] = frequency    

#analysis of correlation outliers
outliers_11ntR = summary_11ntR[(summary_11ntR['R_30Mg'] < 0.85) | (summary_11ntR['R_30Mg'].isna())]


#--------------------------IC3----------------------------------------

#% Get summary of ddGs, and frequencies for natural IC3
data = entire_lib_selected.groupby('r_seq')
wt_data = data.get_group('UAUGG_CCUAAG')
wt_data = wt_data.set_index('old_idx')
scaffolds = wt_data.index
list_IC3 = list(set(details_IC3['receptor_sequence']))
summary_IC3 = pd.DataFrame(index = list_IC3)

ddG_30Mg = []
ddG_5Mg = []
ddG_5Mg150K = [] 

std_30Mg = []
std_5Mg = []
std_5Mg150K = []

R_30Mg = []
R_5Mg = []
R_5Mg150K = []


n_30Mg = []
n_5Mg = []
n_5Mg150K = []

n_R_30Mg = []
n_R_5Mg = []
n_R_5Mg150K = []

r = []
r_5Mg = []
r_5Mg150K = []


r_low = []
r_low_5Mg = []
r_low_5Mg150K = []


r_high = []
r_high_5Mg = []
r_high_5Mg150K = []


n_compared = []
n_compared_5Mg = []
n_compared_5Mg150K = []

for sequences in summary_IC3.index:
    next_data = data.get_group(sequences)
    next_data = next_data.set_index('old_idx')
    next_data = next_data.reindex(scaffolds)
    
    
    ddG = next_data['dG_Mut2_GAAA'].subtract(wt_data['dG_Mut2_GAAA'])
    ddG_30Mg.append(ddG.median())
    std_30Mg.append(ddG.std())
    n_30Mg.append(len(next_data['dG_Mut2_GAAA'].dropna()))
    
    ddG = next_data['dG_Mut2_GAAA_5mM_2'].subtract(wt_data['dG_Mut2_GAAA_5mM_2'])
    ddG_5Mg.append(ddG.median())
    std_5Mg.append(ddG.std())
    n_5Mg.append(len(next_data['dG_Mut2_GAAA_5mM_2'].dropna()))    
    
    ddG = next_data['dG_Mut2_GAAA_5mM_150mMK_1'].subtract(wt_data['dG_Mut2_GAAA_5mM_150mMK_1'])
    ddG_5Mg150K.append(ddG.median())
    std_5Mg150K.append(ddG.std())
    n_5Mg150K.append(len(next_data['dG_Mut2_GAAA_5mM_150mMK_1'].dropna()))      
    
    
    
    
    #for calculating R coefficient only take values that are within threshold
    x = next_data['dG_Mut2_GAAA'].copy()
    x[x>dG_thr] = np.nan
    y = wt_data['dG_Mut2_GAAA'].copy()
    y[y>dG_thr] = np.nan
    R_data = pd.DataFrame()
    R_data['x'] = x
    R_data['y'] = y
    R_data = R_data.dropna()
    n_R_30Mg.append(len(R_data))
    corr = R_data.corr()
    R_30Mg.append(corr['x']['y'])
    #Calculate errors for R values 
    x_data = R_data['x']
    y_data = R_data['y']
    r_,r_low_,r_high_,n_compared_ = Pearson_and_confidence(x_data,y_data)
    r.append(r_)
    r_low.append(r_low_)
    r_high.append(r_high_)
    n_compared.append(n_compared_)
    
    #repeat at 5 mM  Mg
    x = next_data['dG_Mut2_GAAA_5mM_2'].copy()
    x[x>dG_thr] = np.nan
    y = wt_data['dG_Mut2_GAAA'].copy()
    y[y>dG_thr] = np.nan
    R_data = pd.DataFrame()
    R_data['x'] = x
    R_data['y'] = y
    R_data = R_data.dropna()
    n_R_5Mg.append(len(R_data))
    corr = R_data.corr()
    R_5Mg.append(corr['x']['y']) 
    #Calculate errors for R values 
    x_data = R_data['x']
    y_data = R_data['y']
    r_,r_low_,r_high_,n_compared_ = Pearson_and_confidence(x_data,y_data)
    r_5Mg.append(r_)
    r_low_5Mg.append(r_low_)
    r_high_5Mg.append(r_high_)
    n_compared_5Mg.append(n_compared_)   
    
    
    #repeat at 5 mM  Mg + 150 K
    x = next_data['dG_Mut2_GAAA_5mM_150mMK_1'].copy()
    x[x>dG_thr] = np.nan
    y = wt_data['dG_Mut2_GAAA_5mM_150mMK_1'].copy()
    y[y>dG_thr] = np.nan
    R_data = pd.DataFrame()
    R_data['x'] = x
    R_data['y'] = y
    R_data = R_data.dropna()
    n_R_5Mg150K.append(len(R_data))
    corr = R_data.corr()
    R_5Mg150K.append(corr['x']['y']) 
    #Calculate errors for R values 
    x_data = R_data['x']
    y_data = R_data['y']
    r_,r_low_,r_high_,n_compared_ = Pearson_and_confidence(x_data,y_data)
    r_5Mg150K.append(r_)
    r_low_5Mg150K.append(r_low_)
    r_high_5Mg150K.append(r_high_)
    n_compared_5Mg150K.append(n_compared_)     
    
    
summary_IC3['ddG_30Mg'] = ddG_30Mg    
summary_IC3['std_30Mg'] = std_30Mg  
summary_IC3['n_30Mg'] = n_30Mg  
summary_IC3['R_30Mg'] = R_30Mg  
summary_IC3['n_R_30Mg'] = n_R_30Mg  
summary_IC3['r'] = r 
summary_IC3['r_low'] = r_low 
summary_IC3['r_high'] = r_high
summary_IC3['n_compared'] = n_compared


#fill up summary for 5 mM Mg 
summary_IC3['ddG_5Mg'] = ddG_5Mg    
summary_IC3['std_5Mg'] = std_5Mg  
summary_IC3['n_5Mg'] = n_5Mg  
summary_IC3['R_5Mg'] = R_5Mg  
summary_IC3['n_R_5Mg'] = n_R_5Mg  
summary_IC3['r_5Mg'] = r_5Mg
summary_IC3['r_low_5Mg'] = r_low_5Mg 
summary_IC3['r_high_5Mg'] = r_high_5Mg
summary_IC3['n_compared_5Mg'] = n_compared_5Mg

#fill up summary for 5 mM Mg + 150 mM K
summary_IC3['ddG_5Mg150K'] = ddG_5Mg150K    
summary_IC3['std_5Mg150K'] = std_5Mg150K  
summary_IC3['n_5Mg150K'] = n_5Mg150K  
summary_IC3['R_5Mg150K'] = R_5Mg150K  
summary_IC3['n_R_5Mg150K'] = n_R_5Mg150K  
summary_IC3['r_5Mg150K'] = r_5Mg150K
summary_IC3['r_low_5Mg150K'] = r_low_5Mg150K 
summary_IC3['r_high_5Mg150K'] = r_high_5Mg150K
summary_IC3['n_compared_5Mg150K'] = n_compared_5Mg150K

natural_IC3 = details_IC3.groupby('receptor_sequence')
frequency = []
for sequence in summary_IC3.index:
    frequency.append(len(natural_IC3.get_group(sequence)))
summary_IC3['frequency'] = frequency 


#--------------------------VC2----------------------------------------

#% Get summary of ddGs, and frequencies for natural IC3
data = entire_lib_selected.groupby('r_seq')
wt_data = data.get_group('UAUGG_CCUAAG')
wt_data = wt_data.set_index('old_idx')
scaffolds = wt_data.index
list_VC2 = list(set(details_VC2['receptor_sequence']))
summary_VC2 = pd.DataFrame(index = list_VC2)

ddG_30Mg = []
ddG_5Mg = []
ddG_5Mg150K = [] 

std_30Mg = []
std_5Mg = []
std_5Mg150K = []

R_30Mg = []
R_5Mg = []
R_5Mg150K = []


n_30Mg = []
n_5Mg = []
n_5Mg150K = []

n_R_30Mg = []
n_R_5Mg = []
n_R_5Mg150K = []

r = []
r_5Mg = []
r_5Mg150K = []


r_low = []
r_low_5Mg = []
r_low_5Mg150K = []


r_high = []
r_high_5Mg = []
r_high_5Mg150K = []


n_compared = []
n_compared_5Mg = []
n_compared_5Mg150K = []

for sequences in summary_VC2.index:
    next_data = data.get_group(sequences)
    next_data = next_data.set_index('old_idx')
    next_data = next_data.reindex(scaffolds)
    
    ddG = next_data['dG_Mut2_GAAA'].subtract(wt_data['dG_Mut2_GAAA'])
    ddG_30Mg.append(ddG.median())
    std_30Mg.append(ddG.std())
    n_30Mg.append(len(next_data['dG_Mut2_GAAA'].dropna()))

    ddG = next_data['dG_Mut2_GAAA_5mM_2'].subtract(wt_data['dG_Mut2_GAAA_5mM_2'])
    ddG_5Mg.append(ddG.median())
    std_5Mg.append(ddG.std())
    n_5Mg.append(len(next_data['dG_Mut2_GAAA_5mM_2'].dropna()))    
    
    ddG = next_data['dG_Mut2_GAAA_5mM_150mMK_1'].subtract(wt_data['dG_Mut2_GAAA_5mM_150mMK_1'])
    ddG_5Mg150K.append(ddG.median())
    std_5Mg150K.append(ddG.std())
    n_5Mg150K.append(len(next_data['dG_Mut2_GAAA_5mM_150mMK_1'].dropna()))     
    
    
    #for calculating R coefficient only take values that are within threshold
    x = next_data['dG_Mut2_GAAA'].copy()
    x[x>dG_thr] = np.nan
    y = wt_data['dG_Mut2_GAAA'].copy()
    y[y>dG_thr] = np.nan
    R_data = pd.DataFrame()
    R_data['x'] = x
    R_data['y'] = y
    R_data = R_data.dropna()
    n_R_30Mg.append(len(R_data))
    corr = R_data.corr()
    R_30Mg.append(corr['x']['y'])
    #Calculate errors for R values 
    x_data = R_data['x']
    y_data = R_data['y']
    r_,r_low_,r_high_,n_compared_ = Pearson_and_confidence(x_data,y_data)
    r.append(r_)
    r_low.append(r_low_)
    r_high.append(r_high_)
    n_compared.append(n_compared_)
     
    #repeat at 5 mM  Mg
    x = next_data['dG_Mut2_GAAA_5mM_2'].copy()
    x[x>dG_thr] = np.nan
    y = wt_data['dG_Mut2_GAAA'].copy()
    y[y>dG_thr] = np.nan
    R_data = pd.DataFrame()
    R_data['x'] = x
    R_data['y'] = y
    R_data = R_data.dropna()
    n_R_5Mg.append(len(R_data))
    corr = R_data.corr()
    R_5Mg.append(corr['x']['y']) 
    #Calculate errors for R values 
    x_data = R_data['x']
    y_data = R_data['y']
    r_,r_low_,r_high_,n_compared_ = Pearson_and_confidence(x_data,y_data)
    r_5Mg.append(r_)
    r_low_5Mg.append(r_low_)
    r_high_5Mg.append(r_high_)
    n_compared_5Mg.append(n_compared_)   
    
    #repeat at 5 mM  Mg + 150 K
    x = next_data['dG_Mut2_GAAA_5mM_150mMK_1'].copy()
    x[x>dG_thr] = np.nan
    y = wt_data['dG_Mut2_GAAA_5mM_150mMK_1'].copy()
    y[y>dG_thr] = np.nan
    R_data = pd.DataFrame()
    R_data['x'] = x
    R_data['y'] = y
    R_data = R_data.dropna()
    n_R_5Mg150K.append(len(R_data))
    corr = R_data.corr()
    R_5Mg150K.append(corr['x']['y']) 
    #Calculate errors for R values 
    x_data = R_data['x']
    y_data = R_data['y']
    r_,r_low_,r_high_,n_compared_ = Pearson_and_confidence(x_data,y_data)
    r_5Mg150K.append(r_)
    r_low_5Mg150K.append(r_low_)
    r_high_5Mg150K.append(r_high_)
    n_compared_5Mg150K.append(n_compared_)     
         
summary_VC2['ddG_30Mg'] = ddG_30Mg    
summary_VC2['std_30Mg'] = std_30Mg  
summary_VC2['n_30Mg'] = n_30Mg  
summary_VC2['R_30Mg'] = R_30Mg  
summary_VC2['n_R_30Mg'] = n_R_30Mg  
summary_VC2['r'] = r 
summary_VC2['r_low'] = r_low 
summary_VC2['r_high'] = r_high
summary_VC2['n_compared'] = n_compared

#fill up summary for 5 mM Mg 
summary_VC2['ddG_5Mg'] = ddG_5Mg    
summary_VC2['std_5Mg'] = std_5Mg  
summary_VC2['n_5Mg'] = n_5Mg  
summary_VC2['R_5Mg'] = R_5Mg  
summary_VC2['n_R_5Mg'] = n_R_5Mg  
summary_VC2['r_5Mg'] = r_5Mg
summary_VC2['r_low_5Mg'] = r_low_5Mg 
summary_VC2['r_high_5Mg'] = r_high_5Mg
summary_VC2['n_compared_5Mg'] = n_compared_5Mg

#fill up summary for 5 mM Mg + 150 mM K
summary_VC2['ddG_5Mg150K'] = ddG_5Mg150K    
summary_VC2['std_5Mg150K'] = std_5Mg150K  
summary_VC2['n_5Mg150K'] = n_5Mg150K  
summary_VC2['R_5Mg150K'] = R_5Mg150K  
summary_VC2['n_R_5Mg150K'] = n_R_5Mg150K  
summary_VC2['r_5Mg150K'] = r_5Mg150K
summary_VC2['r_low_5Mg150K'] = r_low_5Mg150K 
summary_VC2['r_high_5Mg150K'] = r_high_5Mg150K
summary_VC2['n_compared_5Mg150K'] = n_compared_5Mg150K

natural_VC2 = details_VC2.groupby('receptor_sequence')
frequency = []
for sequence in summary_VC2.index:
    frequency.append(len(natural_VC2.get_group(sequence)))
summary_VC2['frequency'] = frequency 
#%%
#filter data 
#-------------------
n_minimum = 20
#-------------------

summary_11ntR_filt_30Mg = summary_11ntR[summary_11ntR['n_compared'] >= n_minimum]
summary_11ntR_filt_5Mg = summary_11ntR[summary_11ntR['n_compared_5Mg'] >= n_minimum]
summary_IC3_filt = summary_IC3[summary_IC3['n_compared'] >= n_minimum]
summary_VC2_filt = summary_VC2[summary_VC2['n_compared'] >= n_minimum]

#plot using conventional pearson
plt.figure()
a = summary_11ntR_filt_30Mg['ddG_30Mg']
b = summary_11ntR_filt_30Mg['R_30Mg']

#a = summary_11ntR_filt_5Mg['ddG_5Mg']
#b = summary_11ntR_filt_5Mg['R_5Mg']
plt.scatter(a,b,s=60,edgecolors='k',marker='o',color = 'blue')


a = summary_IC3_filt['ddG_30Mg']
b = summary_IC3_filt['R_30Mg']
plt.scatter(a,b,s=120,edgecolors='k',marker='s',color = 'red')


a = summary_VC2_filt['ddG_30Mg']
b = summary_VC2_filt['R_30Mg']
plt.scatter(a,b,s=120,edgecolors='k',marker='s',color = 'purple')

plt.ylim([0,1.2])
plt.title('R_pearson vs ddG of natural variants')
plt.show()

#
#%------------------------------------------------------------
#plot with symbols proportional to the number found in nature
plt.figure()
fig_name = 'r_sq_vs_ddG_select_natura.svg'

for sequence in summary_11ntR_filt_30Mg.index:
    a = summary_11ntR_filt_30Mg.loc[sequence]['ddG_30Mg']
    b = summary_11ntR_filt_30Mg.loc[sequence]['R_30Mg'] **2 
    plt.scatter(a,b,s = 20 * summary_11ntR_filt_30Mg.loc[sequence]['frequency'] ** 0.5 , edgecolors = 'k', marker = 'o', color = 'blue')


for sequence in summary_IC3_filt.index:
    a = summary_IC3_filt.loc[sequence]['ddG_30Mg']
    b = summary_IC3_filt.loc[sequence]['R_30Mg'] ** 2
    plt.scatter(a,b,s = 20 * summary_IC3_filt.loc[sequence]['frequency'] ** 0.5 , edgecolors = 'k', marker = 'o', color = 'red')
    

for sequence in summary_VC2_filt.index:
    a = summary_VC2_filt.loc[sequence]['ddG_30Mg']
    b = summary_VC2_filt.loc[sequence]['R_30Mg'] ** 2
    plt.scatter(a,b,s = 20 * summary_VC2_filt.loc[sequence]['frequency'] ** 0.5 , edgecolors = 'k', marker = 'o', color = 'green')
    
plt.ylim([0,1.2])
plt.xlim([-0.5,4.5])

left,right = plt.xlim()
plt.plot([left,right],[1,1],'k--')
plt.savefig(new_data_path + fig_name)

plt.ylabel('R**2')
plt.title('R^2 vs ddG, symbols prop. to frequency')
plt.show()   

#%----------------------------------------------------------
#plot with symbols all with the same size
plt.figure()
fig_name = 'r_sq_vs_ddG_select_natura_2.svg'

for sequence in summary_11ntR_filt_30Mg.index:
    a = summary_11ntR_filt_30Mg.loc[sequence]['ddG_30Mg']
    b = summary_11ntR_filt_30Mg.loc[sequence]['R_30Mg'] **2 
    plt.scatter(a,b, edgecolors = 'k', marker = 'o', color = 'blue')


for sequence in summary_IC3_filt.index:
    a = summary_IC3_filt.loc[sequence]['ddG_30Mg']
    b = summary_IC3_filt.loc[sequence]['R_30Mg'] ** 2
    plt.scatter(a,b, edgecolors = 'k', marker = 'o', color = 'red')
    

for sequence in summary_VC2_filt.index:
    a = summary_VC2_filt.loc[sequence]['ddG_30Mg']
    b = summary_VC2_filt.loc[sequence]['R_30Mg'] ** 2
    plt.scatter(a,b, edgecolors = 'k', marker = 'o', color = 'green')
    
plt.ylim([0,1.2])
plt.xlim([-0.5,4.5])

left,right = plt.xlim()
plt.plot([left,right],[1,1],'k--')
plt.savefig(new_data_path + fig_name)

plt.ylabel('R**2')
plt.title('R^2 vs ddG')
plt.show() 
#%-------------------------------------------------------------------------

#plot with symbols proportional to the number of measurements made
plt.figure()
for sequence in summary_11ntR_filt_30Mg.index:
    a = summary_11ntR_filt_30Mg.loc[sequence]['ddG_30Mg']
    b = summary_11ntR_filt_30Mg.loc[sequence]['R_30Mg']
    plt.scatter(a,b,s = 20 * summary_11ntR_filt_30Mg.loc[sequence]['n_compared'] ** 0.5 , edgecolors = 'k', marker = 'o', color = 'blue')


for sequence in summary_IC3_filt.index:
    a = summary_IC3_filt.loc[sequence]['ddG_30Mg']
    b = summary_IC3_filt.loc[sequence]['R_30Mg']
    plt.scatter(a,b,s = 20 * summary_IC3_filt.loc[sequence]['n_compared'] ** 0.5 , edgecolors = 'k', marker = 'o', color = 'red')
    

for sequence in summary_VC2_filt.index:
    a = summary_VC2_filt.loc[sequence]['ddG_30Mg']
    b = summary_VC2_filt.loc[sequence]['R_30Mg']
    plt.scatter(a,b,s = 20 * summary_VC2_filt.loc[sequence]['n_compared'] ** 0.5 , edgecolors = 'k', marker = 'o', color = 'green')
    
plt.ylim([0,1.2])
plt.xlim([-0.5,4.5])

left,right = plt.xlim()
plt.plot([left,right],[1,1],'k--')
plt.title('R^2 vs ddG, symbols prop. to n used for correlation')
plt.show()   

#%% plot frequency vs. ddG 
freq = summary_11ntR_filt_30Mg['frequency'].copy()
freq_log = freq.apply('log10')
ddG_30Mg = summary_11ntR_filt_30Mg['ddG_30Mg'].copy()
ddG_5Mg = summary_11ntR_filt_30Mg['ddG_5Mg'].copy()
ddG_5Mg150K = summary_11ntR_filt_30Mg['ddG_5Mg150K'].copy()

plt.figure()
plt.scatter(freq_log,ddG_30Mg)
print('correlation at 30 mM Mg' + str(freq_log.corr(ddG_30Mg)))
plt.title('frequency vs ddG at 30 mM Mg')
plt.show()

print('correlation at 5mM Mg' + str(freq_log.corr(ddG_5Mg)))

plt.figure()
plt.scatter(freq_log,ddG_5Mg150K)
print('correlation at 5mM Mg + 150 mM K' + str(freq_log.corr(ddG_5Mg150K)))
plt.title('frequency vs ddG at 5 mM Mg + 150 mM K')
plt.show()

#%%
fig = plt.figure()
sns.regplot(freq_log,ddG_30Mg,color='blue', scatter_kws={'s':120,'edgecolors':'black'})
plt.scatter(summary_IC3_filt['frequency'].apply('log10'),summary_IC3_filt['ddG_30Mg'],s = 120, color = 'red',edgecolor = 'black')
plt.scatter(summary_VC2_filt['frequency'].apply('log10'),summary_VC2_filt['ddG_30Mg'],s = 120, color = 'cyan',edgecolor = 'black')
plt.ylim([-1,5])
#plt.axes().set_aspect('equal')
adjustFigAspect(fig,aspect=0.9)
plt.savefig(new_data_path + 'frequency_vs_ddG_30Mg.svg')
plt.show()


#%
fig = plt.figure()
sns.regplot(freq_log,ddG_5Mg,color='blue', scatter_kws={'s':120,'edgecolors':'black'})
plt.scatter(summary_IC3_filt['frequency'].apply('log10'),summary_IC3_filt['ddG_5Mg'],s = 120, color = 'red',edgecolor = 'black')
plt.scatter(summary_VC2_filt['frequency'].apply('log10'),summary_VC2_filt['ddG_5Mg'],s = 120, color = 'cyan',edgecolor = 'black')
plt.ylim([-1,5])
#plt.axes().set_aspect('equal')
adjustFigAspect(fig,aspect=0.9)
plt.savefig(new_data_path + 'frequency_vs_ddG_5Mg.svg')
plt.show()

#%
fig = plt.figure()
sns.regplot(freq_log,ddG_5Mg150K,color='blue', scatter_kws={'s':120,'edgecolors':'black'})
plt.scatter(summary_IC3_filt['frequency'].apply('log10'),summary_IC3_filt['ddG_5Mg150K'],s = 120, color = 'red',edgecolor = 'black')
plt.scatter(summary_VC2_filt['frequency'].apply('log10'),summary_VC2_filt['ddG_5Mg150K'],s = 120, color = 'cyan',edgecolor = 'black')
plt.ylim([-1,5])
#plt.axes().set_aspect('equal')
adjustFigAspect(fig,aspect=0.9)
#ax = fig.add_subplot(111)
plt.savefig(new_data_path + 'frequency_vs_ddG_5Mg150K.svg')
plt.show()
