#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:56:56 2020

@author: dsg
"""


import pandas
import numpy

"""
  pandas have 3 basic data types
  series => 1D
  dataframe => 2D
  panel => 3D
"""

data = numpy.random.randint(10,50,(15,4))
cols = ['temp', 'humidity', 'pressure', 'airq']
dates = pandas.date_range('20190725', periods=15)

df = pandas.DataFrame(data, columns = cols, index = dates)
df['temp'] # Accessing a column
df['2019-07-31':'2019-08-03']['temp']
df['2019-07-31':'2019-08-03']['temp'][df['temp']>30] # Get all temps which is greater than 30
stats = df.describe()  # Complete statistical data

# Import data from CSV
with_header_data = pandas.read_csv('/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/datawh.csv')
without_header_data = pandas.read_csv('/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/datanh.csv', header = None)
without_header_data.columns = ['C1', 'C2', 'C3', 'C4']
temps = with_header_data['Temperature'] # Extracts all the temps into a pandas Series (1D)
new_data = with_header_data[['Temperature', 'Pressure']]
location_specific_data = with_header_data.iloc[:, 0:3] # All rows of first three columns
location_specific_data1 = with_header_data.iloc[0:5, 0:3] # First 5 rows, first 3 columns

random_data_without_column = numpy.random.randint(0, 100, (10, 100))
data_frame = pandas.DataFrame(random_data_without_column)
data_frame.columns = ['F'+str(i) for i in range(100)]

# Missing values handling
missing_values = pandas.read_csv('/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/datawh_missing.csv', na_values=['.','*'])
missing_values.isnull().sum() # Check for missing values

drop_missing = missing_values.dropna() # Drop all roes with missing values
# missing_values.dropna(inplace=True) # This will overwrite the dataframe

missing_values.fillna(drop_missing.mean()) # Replace missing with mean everywhere
missing_values['Pressure'].fillna(missing_values['Pressure'].median()) # Replace missing pressures with pressures median
missing_values.dropna(thresh=2, inplace=True) # If more than 2 columns are missing or 0
missing_values.reset_index(inplace=True) # Reset the index
missing_values.drop(['index'], axis=1, inplace=True) # Drop the old index column

missing_values_copy = missing_values.copy()  # Create a copy of original data

# Read data from web source
web_data = pandas.read_html(r"https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130429&end=20200326")
bitcoin = web_data[2]
bitcoin.describe()

# Data grouping 
regiment_scores = pandas.read_csv('/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/regiment.csv')
regiment_scores.drop('index', axis=1, inplace=True)
regiment_scores.describe()
regiment_scores.columns
regiment_scores.regiment.unique()
regiment_scores.groupby(by='regiment')['postTestScore'].mean()
regiment_scores.groupby(by='regiment').describe()
regiment_scores.groupby(by=['regiment', 'company']).describe()
regiment_scores[(regiment_scores["regiment"] == "Scouts") | (regiment_scores["regiment"] == "Dragoons")]

# Statistics
regiment_scores['learning'] = regiment_scores['postTestScore'] - regiment_scores['preTestScore']
regiment_scores['learning'].min()
regiment_scores['learning'].max()
regiment_scores['learning'].std()
regiment_scores['learning'].var()
regiment_scores['learning'].median()
regiment_scores['learning'].mean()
regiment_scores['learning'].mode()
regiment_scores['learning'].skew()
regiment_scores['learning'].kurt()

# Transform a column using lambda function
regiment_scores['growth'] = regiment_scores['learning'].transform(lambda l:l/regiment_scores['learning'].std())


