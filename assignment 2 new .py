#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 22:50:02 2018

@author: noviayou
"""

#predict operational profit

import statsmodels.api as sm
import pandas as pd
import numpy as np

df = pd.read_csv('link.csv')
#df
newdf= df[df.columns[df.isnull().sum()/df.shape[0]<0.7]]
#newdf = df.loc[:, missing_percentage < 0.7]


#newdf
newdf1 = newdf.select_dtypes(include = ['number'])
newdf2 = newdf1.fillna(newdf1.median())
newdf2.head(3)
dfx = newdf2.drop('ebit',axis = 1,)

y = newdf2.ebit

#inplace = True  ---> dataframe 中还是有na
#"oibdp"





def stepwise(X, y, 
            initial_list=[], 
            threshold_in=0.01, 
            threshold_out = 0.05, 
            verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]  #p-value of column in excluded
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

result = stepwise(dfx, y)


print(result)



##assignment 2

def step(x,y,
         initial_list = [],
         critical_value = 0.05,
         candidate_list = []):
    include = list(initial_list)
    new_pval = pd.Series(index=candidate_list)
    
    for columns in candidate_list:
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[include+[columns]]))).fit()
        new_pval[columns] = model.pvalues[columns]
    best_pval = new_pval.min() 
    if best_pval < critical_value: 
         best_feature = new_pval.argmin()
         include.append(best_feature)
         step(x,y,
              initial_list = [],
              critical_value = 0.05,
              candidate_list = [])
    if best_pval > critical_value:
        break
    return include
    

         
    




