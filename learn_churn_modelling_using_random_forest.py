#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:25:17 2020

@author: dsg
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import metrics

data = pd.read_csv(r'/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/churn_modelling.csv')
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
ydata = data['Exited']
xdata = data.drop('Exited', axis=1)

# We will transform Gender and Geograpghy into numeric form
le1 = LabelEncoder()
le1.fit(xdata['Geography'])
xdata['Geography'] = le1.transform(xdata['Geography'])

le2 = LabelEncoder()
le2.fit(xdata['Gender'])
xdata['Gender'] = le2.transform(xdata['Gender'])

# Transforming the Geography data made it a continous feature, it was initially a categorical feature
# We will transform it again into categorical feature using three columns

ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')   
xdata = ct.fit_transform(xdata) # Fit and transform at once

xtr, xts, ytr, yts = train_test_split(xdata, ydata, test_size = 0.2)

# Scale the data
sc = StandardScaler()
sc.fit(xtr)
xtr = sc.transform(xtr)
xts = sc.transform(xts)

alg = ensemble.RandomForestClassifier(n_estimators=3)
alg.fit(xtr, ytr)

train_accuracy = alg.score(xtr, ytr)
test_accuracy = alg.score(xts, yts)

ypred = alg.predict(xts)
metrics.recall_score(yts, ypred) 