#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:11:42 2018

@author: Josh Pause
"""

# load required libs
import pandas as pd
import numpy as np

# import our csv into a dataframe
df = pd.read_csv('data/201710-CAH_PulseOfTheNation.csv')

# keep only desired columns
df = df.iloc[:,[0,1,2,4,5,6,8,10,13,16,18,20,21,22,23,24,25]]
df = df.fillna(0)

print(df.head())

d = {'income':[],'wn_count':[]}
for i in np.arange(int(min(df["Income"])), int(max(df["Income"])),100):
    d["income"].append(i)
    d["wn_count"].append(len(df.loc[(df["Income"]<=i) & (df['From what you have heard or seen, do you mostly agree or mostly disagree with the beliefs of White Nationalists?']=='Agree')]))
    print(i)
    
d = pd.DataFrame(d)    

