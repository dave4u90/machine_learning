#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:43:25 2020

@author: dsg
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import tree


data = pd.read_csv(r'/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/titanic/train.csv')


# Check for missing values
data.isnull().sum()

# Fill missing values
# Feature engeering tip: Divide into two groups from target and then fill for Age
data.Age.fillna(data.groupby("Survived").Age.transform('mean'), inplace=True) 
data.Embarked.fillna(data.Embarked.mode()[0], inplace=True)
data['Family'] = data.SibSp + data.Parch
data['HasCabin'] = ~data['Cabin'].isnull()

# Drop irrelevant features
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)

# =============================================================================
# # Feature engineering analysis
# 
# # Impact of passenger class
# plt.figure(figsize=(12,3))
# sns.countplot(data.Pclass)
# plt.show()
# 
# plt.figure(figsize=(12,3))
# sns.countplot(data.Pclass[data.Survived == 1])
# plt.show()
# 
# # Impact of sex
# plt.figure(figsize=(12,3))
# sns.countplot(data.Sex)
# plt.show()
# 
# plt.figure(figsize=(12,3))
# sns.countplot(data.Sex[data.Survived == 1])
# plt.show()
# 
# # Impact of sibling count
# plt.figure(figsize=(12,3))
# sns.countplot(data.SibSp)
# plt.show()
# 
# plt.figure(figsize=(12,3))
# sns.countplot(data.SibSp[data.Survived == 1])
# plt.show()
# 
# # Impact of parent count
# plt.figure(figsize=(12,3))
# sns.countplot(data.Parch)
# plt.show()
# 
# plt.figure(figsize=(12,3))
# sns.countplot(data.Parch[data.Survived == 1])
# plt.show()
# 
# # Impact of embarkment
# plt.figure(figsize=(12,3))
# sns.countplot(data.Embarked)
# plt.show()
# 
# plt.figure(figsize=(12,3))
# sns.countplot(data.Embarked[data.Survived == 1])
# plt.show()
# 
# # Impact of age
# plt.figure(figsize=(12,3))
# sns.distplot(data.Age[data.Survived == 1])
# sns.distplot(data.Age[data.Survived == 0])
# plt.legend(['1','0'])
# plt.show()
# 
# # Impact of fare
# plt.figure(figsize=(12,3))
# sns.distplot(data.Fare[data.Survived == 1])
# sns.distplot(data.Fare[data.Survived == 0])
# plt.legend(['1','0'])
# plt.show()
# =============================================================================

# Encode categorical labels
le1 = LabelEncoder()
data.Sex = le1.fit_transform(data.Sex)

le2 = LabelEncoder()
data.Embarked = le2.fit_transform(data.Embarked)

xdata = data.drop(['Survived'], axis=1)
ydata = data['Survived']

ct = ColumnTransformer([('Embarked', OneHotEncoder(), [4])], remainder = 'passthrough')   
xdata = ct.fit_transform(xdata)

# train the algorithm
alg = tree.DecisionTreeClassifier(max_depth=6)
alg.fit(xdata, ydata)
alg.score(xdata, ydata)

test_data = pd.read_csv(r'/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/titanic/test.csv')
# Fill missing values
data.Age.fillna(data.groupby("Survived").Age.transform('mean'), inplace=True)
test_data.Fare.fillna(test_data.Fare.median(), inplace=True)
test_data.Embarked.fillna(test_data.Embarked.mode()[0], inplace=True)
test_data['Family'] = test_data.SibSp + test_data.Parch
test_data['HasCabin'] = ~test_data['Cabin'].isnull()

# Drop irrelevant features
test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Cabin'], axis=1, inplace=True)

test_data.Sex = le1.fit_transform(test_data.Sex)
test_data.Embarked = le2.fit_transform(test_data.Embarked)
test_data = ct.fit_transform(test_data)

# Prediction
ypred = alg.predict(test_data)

results = pd.read_csv(r'/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/titanic/test.csv')
results['Survived'] = ypred
results = results[['PassengerId', 'Survived']]
results.to_csv(r'/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/titanic/prediction.csv')