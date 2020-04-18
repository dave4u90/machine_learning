#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:21:00 2020

@author: dsg
"""

# import numpy as np
import pandas as pd
# from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
import sklearn.metrics as mtr 
import seaborn as sns

sns.set(style='whitegrid', color_codes=True)


"""
Attribute Information:

Input variables:
# bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
"""


data = pd.read_csv(r'/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/bank_marketing.csv')

# Barplot for dependent variables
sns.countplot(x='y', data = data, palette = 'hls')
# Can be simply written like the following
# sns.countplot(data['y'])
plt.show()

# Customer Job distribution
plt.figure(figsize=(12,5))
sns.countplot(y = 'job', data = data) # trick, plot job on y axis to get a clean plot
plt.show() 

# Marital status distribution
sns.countplot(data['marital'])
plt.show() 

# Barplot for credit in default
sns.countplot(x='default', data = data)
plt.show()  

# Barplot for housing loan
sns.countplot(x='housing', data = data)
plt.show()  

# Barplot for personal loan
sns.countplot(x='loan', data = data)
plt.show()

# Barplot for previous marketing campaign outcome
sns.countplot(data['poutcome'])
plt.show()

# We will drop unnecessary data 
data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis = 1, inplace = True)

"""
In logistic regression models, encoding of all the independent variables as dummy varibales,
allows easy interpretation and calculation of the odds ratios, and increases the stability
and significance of the coefficients
"""

data2 = pd.get_dummies(data, columns = ['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
data2.drop(data2.columns[[12, 16, 18, 21, 24]], axis = 1, inplace = True)

#  We need to change yes/no to 0/1
data2.y.replace(('yes', 'no'), (1, 0), inplace=True)

# Check independence between independent variables
sns.heatmap(data2.corr())
plt.show()

ydata = data2['y'] 
xdata = data2.drop('y', axis = 1)

xtr, xts, ytr, yts = train_test_split(xdata, ydata, test_size = 0.2)

# Scale the data using a standard scalar
scalar = StandardScaler()
scalar.fit(xtr)
xtr = scalar.transform(xtr)
xts = scalar.transform(xts)

alg = LogisticRegression()
alg.fit(xtr, ytr)

ypred = alg.predict(xts)

# confusion matrix
confusion_matrix = mtr.confusion_matrix(yts, ypred)

# Overall accuracy
accuracy = alg.score(xts, yts)

# Recall score
recall = mtr.recall_score(yts, ypred) 

# Classification Report gives everything
report = mtr.classification_report(yts, ypred)


