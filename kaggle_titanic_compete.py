#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:43:25 2020

@author: dsg
"""


import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn import tree

train_data = pd.read_csv(r'/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/titanic/train.csv')
test_data = pd.read_csv(r'/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/titanic/test.csv')

passenger_data = pd.concat([train_data.drop(['Survived'], axis=1), test_data])

# Now transform the sex column into ordinal
passenger_data.Sex = passenger_data.Sex.transform(lambda x: int(x == 'female') + 1)

passenger_data['Salutation'] = passenger_data.Name.transform(lambda x: x.split(',')[1].split('.')[0].strip())

# Now we do not need name, so drop it
passenger_data.drop(['Name'], axis=1, inplace=True)

# Correct the wrong salutations
passenger_data.loc[~passenger_data.Salutation.isin(['Mr', 'Mrs', 'Miss', 'Ms', 'Master']), 'Salutation'] = passenger_data.loc[~passenger_data.Salutation.isin(['Mr', 'Mrs', 'Miss', 'Ms', 'Master']), 'Sex'].transform(lambda x: 'Mr' if x == 'male' else 'Ms')

# Fill missing ages
passenger_data.Age.fillna(passenger_data.groupby('Salutation').Age.transform('median'), inplace=True)

# Fill missing embarked
passenger_data.Embarked.fillna(passenger_data.Embarked.mode()[0], inplace=True)

# =============================================================================
# # Classify the age into four groups, child, teens, adult, old_age
# 
# def age_order(x):
#     if x <= 12:
#         return 4
#     elif x > 12 and x <= 19:
#         return 3
#     elif x > 19 <= 50:
#         return 2
#     elif x > 50:
#         return 4
#     
# passenger_data['AgeWeight'] = passenger_data.Age.transform(lambda x: age_order(x))
# 
# # Create a new feature family and add ordinal
# passenger_data['Family'] = passenger_data.SibSp + passenger_data.Parch
# def family_order(x):
#     if x == 0:
#         return 4
#     elif x == 1:
#         return 2
#     else:
#         return 1
# 
# passenger_data['FamilyWeight'] = passenger_data.Family.transform(lambda x: family_order(x))
# 
# # Remove all family information
# passenger_data.drop(['Family', 'SibSp', 'Parch'], axis=1, inplace=True)
# 
# # Now we do not need age, so drop it
# passenger_data.drop(['Age'], axis=1, inplace=True)
# 
# =============================================================================
# Only one old age class 3 passenger have missing fare, fillup
passenger_data.Fare.fillna(passenger_data.groupby('Pclass').Fare.transform('median'), inplace=True)
# passenger_data.drop(['Fare'], axis=1, inplace=True)

# Pclass has ordinal property, so encode it as such
# passenger_data.Pclass = passenger_data.Pclass.transform(lambda x: 3 if x == 1 else (1 if x == 3 else 2))

# Now fill up the cabin information from ticket no
passenger_data.Cabin.fillna(passenger_data.groupby('Ticket').Ticket.transform(lambda x: x.mode()[0]), inplace=True)

# Now we do not need ticket, drop it
passenger_data.drop(['Ticket'], axis=1, inplace=True)

# Now convert cabin information to boolean
passenger_data['HasCabin'] = passenger_data.Cabin.isnull().transform(lambda x: 1 if x == False else 0)

# Now we do not need cabin, drop it
passenger_data.drop(['Cabin'], axis=1, inplace=True)

# Now we dont need salutation, drop it
passenger_data.drop(['Salutation'], axis=1, inplace=True)

# Now encode Embarked information
le = LabelEncoder()
ct = ColumnTransformer([('Embarked', OneHotEncoder(), [3])], remainder = 'passthrough')   
passenger_data.Embarked = le.fit_transform(passenger_data.Embarked)

data_to_train = passenger_data[passenger_data.PassengerId.isin(np.arange(1,892))]
data_to_train.drop(['PassengerId'], axis=1, inplace=True)
data_to_train = ct.fit_transform(data_to_train)

data_to_test = passenger_data[passenger_data.PassengerId.isin(np.arange(892,1310))]
data_to_test.drop(['PassengerId'], axis=1, inplace=True)
data_to_test = ct.fit_transform(data_to_test)

# train the algorithm
alg = tree.DecisionTreeClassifier(max_depth=7)
alg.fit(data_to_train, train_data.Survived)
alg.score(data_to_train, train_data.Survived)

# Prediction
survived = alg.predict(data_to_test)

results = pd.read_csv(r'/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/titanic/test.csv')
results['Survived'] = survived
results = results[['PassengerId', 'Survived']]
results.to_csv(r'/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/titanic/prediction.csv')
