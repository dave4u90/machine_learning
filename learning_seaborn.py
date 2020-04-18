#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 20:31:36 2020

@author: dsg
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/churn_modelling.csv")

# Checking whether credit score has to do anything with whether a customer stays or leaves
plt.figure(figsize=(12,5))
sns.distplot(data['CreditScore'][data['Exited'] == 1]) # Distribution plot to show distribution of credit score of lost customers
sns.distplot(data['CreditScore'][data['Exited'] == 0]) # Distribution plot to show distribution of credit score of existing customers
plt.legend(['Lost Customers', 'Existing Customers'])
plt.show()

# Checking whether age has to do anything with whether a customer stays or leaves
plt.figure(figsize=(12,5))
sns.distplot(data['Age'][data['Exited'] == 1]) # Distribution plot to show distribution of credit score of lost customers
sns.distplot(data['Age'][data['Exited'] == 0]) # Distribution plot to show distribution of credit score of existing customers
plt.legend(['Lost Customers', 'Existing Customers'])
plt.show()

# Whether balanceis impacting
plt.figure(figsize=(12,5))
sns.distplot(data['Tenure'][data['Exited'] == 1]) # Distribution plot to show distribution of credit score of lost customers
sns.distplot(data['Tenure'][data['Exited'] == 0]) # Distribution plot to show distribution of credit score of existing customers
plt.legend(['Lost Customers', 'Existing Customers'])
plt.show()


# With a categorical feature
plt.figure(figsize=(12,5))
sns.countplot(data['Geography'])
plt.show()  # Just check the no of customers from three countries

# Now check who are leaving

# With a categorical feature
plt.figure(figsize=(12,5))
sns.countplot(data['Geography'][data['Exited'] == 1])
plt.show()  # Just check the no of customers from three countries

# In the same graph with another categorical feature
plt.figure(figsize=(12,5))
sns.countplot(data['Gender'], hue=data['Exited'])
plt.show()

# Plot correlation by heatmap
corr = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm') # With annotation and coolwarm colors
plt.show()