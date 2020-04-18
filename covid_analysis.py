#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 23:12:24 2020

@author: dsg
"""


import pandas as pd

raw_data = pd.read_html(r"https://docs.google.com/spreadsheets/d/e/2PACX-1vSc_2y5N0I67wDU38DjDh35IZSIS30rQf7_NYZhtYYGU1jJYT6_kDx4YpF-qw0LSlGsBYP8pqM_a1Pd/pubhtml")
covids = raw_data[0]

covids.dropna(inplace=True)
covids.reset_index(inplace=True)
covids.drop('index', axis=1, inplace=True)
# covids.describe()

# Save data to local file
covids.to_excel(r'/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/covids.xlsx')
data = pd.read_excel(r'/home/dsg/Documents/Projects/Machine Learing and Deep Learning/Data/covids.xlsx')

# Read the JSON from API

json_data = pd.read_json(r'https://api.covid19india.org/raw_data.json')
data_frame = pd.DataFrame(json_data['raw_data'])