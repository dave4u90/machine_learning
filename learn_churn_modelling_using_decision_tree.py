#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:39:00 2020

@author: dsg
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
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

alg = tree.DecisionTreeClassifier(max_depth = 100)
alg.fit(xtr, ytr)

train_accuracy = alg.score(xtr, ytr)
test_accuracy = alg.score(xts, yts)

# The different between the train and test accuracy is considerably huge, so the tree is overfitted
# Now let's reduce the depth of the tree, depth is an experiment to be done to match the two accuracies as close as possible
alg2 = tree.DecisionTreeClassifier(max_depth = 8)
alg2.fit(xtr, ytr)

train_accuracy2 = alg2.score(xtr, ytr)
test_accuracy2 = alg2.score(xts, yts)

ypred = alg2.predict(xts)
metrics.recall_score(yts, ypred) # Better than logistic regression

# Now check the confusion matrix 
cm = metrics.confusion_matrix(yts, ypred) # Better than logistic regression

# Classification Report gives everything
report = metrics.classification_report(yts, ypred)

# Now improve the performance even beter
# Check the probability
predprob = alg2.predict_proba(xts)

# Make a dataframe and analyse
pred_dt = pd.DataFrame(predprob)
pred_dt['predicted'] = ypred

# ypred is done by a default probability threshold of 0.5, we will test with miltiple other thresholds to check
# if that can improve the recall
thresh = np.arange(0, 1.05, 0.05)

prob_compare = pd.DataFrame([], columns = ['Probability', 'Recall', 'TP', 'FP', 'TN', 'FN', 'Cost'])

for i in thresh:
    ypred1 = pred_dt[1].transform(lambda x: 1 if x > i else 0)
    cm = metrics.confusion_matrix(yts, ypred1)
    recall = metrics.recall_score(yts, ypred1)
    cost = cm[0][1] + 5*cm[1][0]
    prob_compare = prob_compare.append({
        'Probability': i , 
        'Recall': recall,
        'TP': cm[1][1],
        'FP': cm[0][1],
        'TN': cm[0][0],
        'FN': cm[1][0],
        'Cost': cost
        } , ignore_index=True)
    
# Analysing the prob_compare datadrame it looks like 0.2 probability would be the most cost effective one.

# Decision tree visualization

from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
fn = ['France', 'Germany', 'Spain', 'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
cn=['Not Exited', 'Exited']

from IPython.display import Image

tree.export_graphviz(
    alg,
    out_file=dot_data,
    feature_names=fn,
    class_names=cn,
    filled=True,
    rounded=True,
    impurity=False
)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
# Writing the image is not working because it's too large, copy the image from IPython console and paste in GIMP to view.
#graph.write_png('dtree.png')