#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 18:22:36 2018

@author: noviayou
"""


import pandas as pd

df = pd.read_csv('com.csv')
#df

number_of_observation = df.count()
number_of_missing = df.isnull().sum()
#print(number_of_missing)
df.describe()

frac = len(df) * 0.5
newdf = df.dropna(thresh = frac, axis = 1)
#newdf = df.loc[:,pd.notnull(df).sum()>len(df)*.5]
newdf

Newnumber_of_observation = newdf.count()
Newnumber_of_missing = df.isnull().sum()
#print(Newnumber_of_observation)
#print(Newnumber_of_missing)

Total_Asset = newdf['atq'].describe()

#print(Total_Asset)
Total_Equity = newdf['ceqq'].describe()
Longterm_Debt = newdf['dlttq'].describe()

summ = pd.DataFrame({'Total Asset':Total_Asset,
                     'Total Equity':Total_Equity,
                     'Longterm Debt':Longterm_Debt})

summ 
