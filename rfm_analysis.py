#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 00:26:18 2020

@author: dsg
"""


import pandas as pd


data = pd.read_excel(r"/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/OnlineRetail.xlsx")
data.drop(['InvoiceNo', 'StockCode', 'Description', 'Country'], axis=1, inplace=True)
data.dropna(inplace=True)
max_date = data['InvoiceDate'].max()
data['R'] = data['InvoiceDate'].transform(lambda d:(max_date-d).days)
data['M'] = data['Quantity'] * data['UnitPrice']
aggregated_data = data.groupby('CustomerID')
data['F'] = aggregated_data['R'].transform(lambda x:x.count()) # Group by and then count rows, can be done using any column, here I have taken R
data.drop(['InvoiceDate', 'Quantity', 'UnitPrice'], axis=1, inplace=True)
