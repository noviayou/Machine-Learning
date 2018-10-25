#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 10:31:13 2018

@author: noviayou
"""

import numpy as np
import pandas as pd

d1 = pd.read_stata('cds.dta')
d2 = pd.read_csv('cds1.csv')
d2.columns = map(str.lower, d2.columns)

d1.gvkey = d1.gvkey.astype('int')

d2['mdate'] = d2['datadate'].apply(lambda x: pd.to_datetime(str(x),format = '%Y%m%d'))


full1 = d1.merge(d2,on=['gvkey','mdate'],how = 'left')
#d1 mdate 2004-08-31 monthly
#d2 datadate 20100228 quarterly (2,3,8,11)

full1.isnull().sum()
full1.describe()
len(full1)-full1.count()
