#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 10:31:13 2018

@author: noviayou
"""

import numpy as np
import pandas as pd

d1 = pd.read_stata('cds.dta')
d2 = pd.read_csv('WRDS.csv')
d2.columns = map(str.lower, d2.columns)

d1.gvkey = d1.gvkey.astype('int')

d2['mdate'] = d2['datadate'].apply(lambda x: pd.to_datetime(str(x),format = '%Y%m%d'))

d1['year'] = pd.DatetimeIndex(d1['mdate']).year
d1['month'] = pd.DatetimeIndex(d1['mdate']).month

d2['year'] = pd.DatetimeIndex(d2['mdate']).year
d2['month'] = pd.DatetimeIndex(d2['mdate']).month

d2.month.unique()

#d2['quarter']= d2.apply(lambda x : 1 if d2['month'] == (2,3) else )
#
#def type(i):
#    if d2.month[i] == 1 & d2.month[i] == 4 & d2.month[i] == 7 & d2.month[i] == 10:
#        return 1
#    if d2.month[i] == 2 & d2.month[i] == 5 & d2.month[i] == 8 & d2.month[i] == 11:
#        return 2   
#    if d2.month[i] == 3 & d2.month[i] == 6 & d2.month[i] == 9 & d2.month[i] == 12:
#        return 3
#    return 'Other'
#
#d2['type']= d2.apply(type,axis = 1)

#
#for i in d2.month:
#    if d2.month[i] == 1 & d2.month[i] == 4 & d2.month[i] == 7 & d2.month[i] == 10:
#        d2.type[i] = 1
#    if d2.month[i] == 2 & d2.month[i] == 5 & d2.month[i] == 8 & d2.month[i] == 11:
#        d2.type[i] = 2   
#    if d2.month[i] == 3 & d2.month[i] == 6 & d2.month[i] == 9 & d2.month[i] == 12:
#        d2.type[i] = 3
#    

##np.where
#        
#d2['type']= d2.apply(lambda x : 1 if d2['month'] == 1 else 0,axis = 1)
#
#
#
#
#d2['type']= d2['month'].apply(lambda x : 1 if x == 4 else 0, axis = 1)
#
#d2['type']= d2['month'].apply(lambda x : 1 if x == 7 else x)
#d2['type']= d2['month'].apply(lambda x : 1 if x == 10 else x)
#
#d2['type']= d2['month'].apply(lambda x : 2 if x == 2 else x)
#d2['type']= d2['month'].apply(lambda x : 2 if x == 5 else x)
#d2['type']= d2['month'].apply(lambda x : 2 if x == 8 else x)
#d2['type']= d2['month'].apply(lambda x : 2 if x == 11 else x)
#
#d2['type']= d2['month'].apply(lambda x : 3 if x == 3 else x)
#d2['type']= d2['month'].apply(lambda x : 3 if x == 6 else x)
#d2['type']= d2['month'].apply(lambda x : 3 if x == 9 else x)
#d2['type']= d2['month'].apply(lambda x : 3 if x == 12 else x)
#


Type1 = d2.loc[d2['month'].isin([1,4,7,10])]
Type1 = Type1.assign(Type = 1)
Q11 = Type1.loc[Type1['month'] == 1]
Q11 = Q11.assign(quarter = 11)
Q12 = Type1.loc[Type1['month'] == 4]
Q12 = Q12.assign(quarter = 12)
Q13 = Type1.loc[Type1['month'] == 7]
Q13 = Q13.assign(quarter = 13)
Q14 = Type1.loc[Type1['month'] == 10]
Q14 = Q14.assign(quarter = 14)
newt1 = pd.concat([Q11,Q12,Q13,Q14])


Type2 = d2.loc[d2['month'].isin([2,5,8,11])]
Type2 = Type2.assign(Type = 2)
Q21 = Type2.loc[Type2['month'] == 2]
Q21 = Q21.assign(quarter = 21)
Q22 = Type2.loc[Type2['month'] == 5]
Q22 = Q22.assign(quarter = 22)
Q23 = Type2.loc[Type2['month'] == 8]
Q23 = Q23.assign(quarter = 23)
Q24 = Type2.loc[Type2['month'] == 11]
Q24 = Q24.assign(quarter = 24)
newt2 = pd.concat([Q21,Q22,Q23,Q24])

Type3 = d2.loc[d2['month'].isin([3,6,9,12])]
Type3 = Type3.assign(Type = 3)
Q31 = Type3.loc[Type3['month'] == 3]
Q31 = Q31.assign(quarter = 31)
Q32 = Type3.loc[Type3['month'] == 6]
Q32 = Q32.assign(quarter = 32)
Q33 = Type3.loc[Type3['month'] == 9]
Q33 = Q33.assign(quarter = 33)
Q34 = Type3.loc[Type3['month'] == 12]
Q34 = Q34.assign(quarter = 34)
newt3 = pd.concat([Q31,Q32,Q33,Q34])

newd2 = newt1.append(newt2,ignore_index = True)
newd2 = newd2.append(newt3,ignore_index = True)


d3 = newd2[['gvkey','Type']]
#m1 = d1.merge(newd2, on = 'gvkey')

newd1 = d1.merge(d3,on = 'gvkey',how= 'left')
newd1 = newd1.drop_duplicates()

Typ1 = newd1.loc[newd1['Type'] == 1]
Typ2 = newd1.loc[newd1['Type'] == 2]
Typ3 = newd1.loc[newd1['Type'] == 3]

Typ1['quarter'] = pd.PeriodIndex(Typ1['mdate'],freq='Q-DEC').strftime('1%q')
Typ2['quarter'] = pd.PeriodIndex(Typ2['mdate'],freq='Q-JAN').strftime('2%q')
Typ3['quarter'] = pd.PeriodIndex(Typ3['mdate'],freq='Q-FEB').strftime('3%q')
finald1 = pd.concat([Typ1,Typ2,Typ3])

finald1['quarter'] = finald1['quarter'].astype('str').astype('int64')

#finald1['quarter']=int(finald1['quarter'])

full1 = finald1.merge(newd2,on=['gvkey','year','quarter'],how='left')
#d1 mdate 2004-08-31 monthly
#d2 datadate 20100228 quarterly (2,5,8,11)(1,4,7,10)(3,6,9,12)
#d2 mdate 2010-02-28 quarterly



#d1.merge(d2,left_on=['gvkey','year','quarter'],right_on=['gvkey','year','quarter'])
#dtype

#condition// choices //np.select(condition,choices)


full1.isnull().sum()
full1.describe()
len(full1)-full1.count()


####assignment 5
full1 = full1.fillna(full1.median())
full1 = full1.select_dtypes(include = ['number'])

full1 = full1.dropna(axis = 'columns', how = 'all')

dftest = full1[full1['year'] == 2016]
dftrain = full1[~full1['year'].isin([2016])]

#Y_column = 'spread5y'
Y_train = dftrain['spread5y']
#X_train = dftrain.drop(Y_column,axis = 1)
X_train = dftrain.drop('spread5y',axis = 1)


Y_test = dftest['spread5y']
X_test = dftest.drop('spread5y',axis = 1)

from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 50)
regressor.fit(X_train,Y_train)

pred = regressor.predict(X_test)
errors = abs(pred - Y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2))
print('Mean Accuracy:', regressor.score(X_test,Y_test))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / Y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')



namesX = X_train.columns
# Get numerical feature importances
importances = list(regressor.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(X_train, round(importance, 4)) for X_train, importance in zip(namesX, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
#[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

features = pd.DataFrame(feature_importances)

####
#feature_importances1 = pd.DataFrame(regressor.feature_importances_,
#                                   index = X_train.columns,
#                                   columns = ['importances']).sort_values('importances')
#

newfeatures = features.iloc[0:50,]
newfeatures = newfeatures.iloc[:,0]

newtrain_x = X_train[list(newfeatures)]
newtest_x = X_test[list(newfeatures)]

regressor1 = RandomForestRegressor(n_estimators = 100,max_depth = 3)
regressor1.fit(newtrain_x,Y_train)
pred1 = regressor1.predict(newtest_x)
print('Mean Accuracy1:', regressor1.score(newtest_x,Y_test))
errors1 = abs(pred1 - Y_test)
mape1 = 100 * (errors1 / Y_test)
# Calculate and display accuracy
accuracy1 = 100 - np.mean(mape1)
print('Accuracy1:', round(accuracy1, 2), '%.')

regressor2 = RandomForestRegressor(n_estimators = 200,max_depth = 3)
regressor2.fit(newtrain_x,Y_train)
pred2 = regressor2.predict(newtest_x)
print('Mean Accuracy2:', regressor2.score(newtest_x,Y_test))
errors2 = abs(pred2 - Y_test)
mape2 = 100 * (errors2 / Y_test)
# Calculate and display accuracy
accuracy2 = 100 - np.mean(mape2)
print('Accuracy2:', round(accuracy2, 2), '%.')


regressor3 = RandomForestRegressor(n_estimators = 500,max_depth = 3)
regressor3.fit(newtrain_x,Y_train)
pred3 = regressor3.predict(newtest_x)
print('Mean Accuracy3:', regressor3.score(newtest_x,Y_test))
errors3 = abs(pred3 - Y_test)
mape3 = 100 * (errors3 / Y_test)
# Calculate and display accuracy
accuracy3 = 100 - np.mean(mape3)
print('Accuracy3:', round(accuracy3, 2), '%.')

regressor4 = RandomForestRegressor(n_estimators = 1000,max_depth = 3)
regressor4.fit(newtrain_x,Y_train)
pred4 = regressor4.predict(newtest_x)
print('Mean Accuracy4:', regressor4.score(newtest_x,Y_test))
errors4 = abs(pred4 - Y_test)
mape4 = 100 * (errors4/ Y_test)
# Calculate and display accuracy
accuracy4 = 100 - np.mean(mape4)
print('Accuracy4:', round(accuracy4, 2), '%.')



GB1 = ensemble.GradientBoostingRegressor(n_estimators = 100, max_depth = 3)
GB1.fit(newtrain_x, Y_train)
mse1 = mean_squared_error(Y_test, GB1.predict(newtest_x))
print("MSE1: %.4f" % mse1)

GB2 = ensemble.GradientBoostingRegressor(n_estimators = 200, max_depth = 3)
GB2.fit(newtrain_x, Y_train)
mse2 = mean_squared_error(Y_test, GB2.predict(newtest_x))
print("MSE2: %.4f" % mse2)

GB3 = ensemble.GradientBoostingRegressor(n_estimators = 500, max_depth = 3)
GB3.fit(newtrain_x, Y_train)
mse3 = mean_squared_error(Y_test, GB3.predict(newtest_x))
print("MSE3: %.4f" % mse3)

GB4 = ensemble.GradientBoostingRegressor(n_estimators = 1000, max_depth = 3)
GB4.fit(newtrain_x, Y_train)
mse4 = mean_squared_error(Y_test, GB4.predict(newtest_x))
print("MSE4: %.4f" % mse4)

import xgboost

xgb1 = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
xgb1.fit(newtrain_x, Y_train)
xgbmse1 = mean_squared_error(Y_test, xgb1.predict(newtest_x))
print("xgbMSE1: %.4f" % xgbmse1)

xgb2 = xgboost.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3)
xgb2.fit(newtrain_x, Y_train)
xgbmse2 = mean_squared_error(Y_test, xgb2.predict(newtest_x))
print("xgbMSE2: %.4f" % xgbmse2)

xgb3 = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=3)
xgb3.fit(newtrain_x, Y_train)
xgbmse3 = mean_squared_error(Y_test, xgb3.predict(newtest_x))
print("xgbMSE3: %.4f" % xgbmse3)

xgb4 = xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=3)
xgb4.fit(newtrain_x, Y_train)
xgbmse4 = mean_squared_error(Y_test, xgb4.predict(newtest_x))
print("xgbMSE4: %.4f" % xgbmse4)





