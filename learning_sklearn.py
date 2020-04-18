#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:43:48 2020

@author: dsg
"""

# Real Problem: https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import metrics
"""
/home/dsg/anaconda3/lib/python3.7/site-packages/sklearn/externals/joblib/__init__.py:15: 
FutureWarning: sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
Please import this functionality directly from joblib, which can be installed with: pip install joblib. 
If this warning is raised when loading pickled models, 
you may need to re-serialize those models with scikit-learn 0.21+.
  warnings.warn(msg, category=FutureWarning)
"""
from sklearn.externals import joblib as jl
# import joblib as jl

data = pd.read_excel(r"/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/PowerPlant.xlsx")

corr = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm') # With annotation and coolwarm colors
plt.show()

# Good correlation exists with PE, proceeding with linear regression
ydata = data['PE']
xdata = data.drop('PE', axis=1)

# We will split the dataset, the larger dataset to train the algo, the rest to test the algo. We will be using sklearn

xtr, xts, ytr, yts = tts(xdata, ydata, test_size = 0.2) # 20% data is for test

alg = LinearRegression()

# Equation => y = m1x1 + m2x2 + m3x3 + m4x4 + c

# Train the algo
alg.fit(xtr, ytr)

# Get the m's
alg.coef_

# Get the c
alg.intercept_

# Check accurancy of the model on test data
accuracy = alg.score(xts, yts)
print(accuracy)

ypred = alg.predict(xts)
metrics.r2_score(yts, ypred)

# Verify the result
# We will pass a row of information and then check the PE value as the output
# Expectation is it should almost never equal to the actual PE, but should be very close to that

input = np.array([13.97, 39.16, 1016.05, 84.6]).reshape(1, -1) # Without reshaping the array it will throw error, old version supported without reshaping
alg.predict(input)

# We want to implement the same algo using L1 regression or LASSO
xtr, xts, ytr, yts = tts(xdata, ydata, test_size = 0.3, random_state = 42) # 30% test data
lasso = Lasso(alpha = 0.1, normalize = True)
lasso.fit(xtr, ytr)
lasso_pred = lasso.predict(xts)
lasso.score(xts, yts)

# We want to implement the same algo using L2 regression or Ridge
xtr, xts, ytr, yts = tts(xdata, ydata, test_size = 0.3, random_state = 42) # 30% test data
ridge = Ridge(alpha = 0.1, normalize = True)
ridge.fit(xtr, ytr)
ridge.pred = ridge.predict(xts)
ridge.score(xts, yts)

# Now we will store the trained model as a pkl file
jl.dump(ridge, 'ridge.pkl')


