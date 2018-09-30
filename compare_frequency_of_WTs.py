#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:57:52 2018

@author: Steve
"""

import pandas as pd
import matplotlib.pyplot as plt
#%%
data_path = '/Users/Steve/Desktop/Data_analysis_code/Data/'
WT_11ntR_location = pd.read_csv(data_path + 'WT_11ntR_natural_location.csv',header = 0)
WT_11ntR_found = WT_11ntR_location[WT_11ntR_location['GNRA_found'] == 1]
WT_11ntR_grouped = WT_11ntR_found.groupby('location')
print(WT_11ntR_grouped.agg('count'))
#%%
WT_IC3_location = pd.read_csv(data_path + 'WT_IC3_natural_location.csv',header = 0)
WT_IC3_found = WT_IC3_location[WT_IC3_location['GNRA_found'] == 1]
WT_IC3_grouped = WT_IC3_found.groupby('location')
print(WT_IC3_grouped.agg('count'))
#%%
WT_VC2_location = pd.read_csv(data_path + 'WT_VC2_natural_location.csv',header = 0)
WT_VC2_found = WT_VC2_location[WT_VC2_location['GNRA_found'] == 1]
WT_VC2_grouped = WT_VC2_found.groupby('location')
print(WT_VC2_grouped.agg('count'))
#%%
WC_5bps_location = pd.read_csv(data_path + 'WC_5bps_GUAA_natural_location.csv',header = 0)
WC_5bps_found = WC_5bps_location[WC_5bps_location['GNRA_found'] == 1]
WC_5bps_grouped = WC_5bps_found.groupby('location')
print(WC_5bps_grouped.agg('count'))
#%%
locations = list(set(WT_11ntR_found.location))
locations.remove("'motif'")
WT_location = pd.DataFrame(index = locations)
#%%
wt_11ntR_count = []
wt_IC3_count = []
wt_VC2_count = []
wc_5bps_count = []
for location in locations:
    if location in list(WT_11ntR_found.location):
        wt_11ntR_count.append(WT_11ntR_grouped.get_group(location)['GNRA_found'].count())
    else:
        wt_11ntR_count.append(0)
        
    if location in list(WT_IC3_found.location):
        wt_IC3_count.append(WT_IC3_grouped.get_group(location)['GNRA_found'].count())
    else:
        wt_IC3_count.append(0)        
 
    if location in list(WT_VC2_found.location):
        wt_VC2_count.append(WT_VC2_grouped.get_group(location)['GNRA_found'].count())
    else:
        wt_VC2_count.append(0)        
        
    if location in list(WC_5bps_found.location):
        wc_5bps_count.append(WC_5bps_grouped.get_group(location)['GNRA_found'].count())
    else:
        wc_5bps_count.append(0)     
    
WT_location['11ntR'] = wt_11ntR_count
WT_location['11ntR_%'] = WT_location['11ntR']/WT_location['11ntR'].sum()
WT_location['IC3'] = wt_IC3_count
WT_location['IC3_%'] = WT_location['IC3']/WT_location['IC3'].sum()
WT_location['VC2'] = wt_VC2_count
WT_location['VC2_%'] = WT_location['VC2']/WT_location['VC2'].sum()
WT_location['GUAA_5bps'] = wc_5bps_count
WT_location['GUAA_5bps_%'] = WT_location['GUAA_5bps']/WT_location['GUAA_5bps'].sum()

WT_location[['11ntR_%','IC3_%','VC2_%','GUAA_5bps_%']].plot.bar()
plt.figure()
WT_location[['11ntR','IC3','VC2','GUAA_5bps']].plot.bar(logy=True)
plt.set_yscale('log')
