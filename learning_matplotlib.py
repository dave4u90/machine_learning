#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 20:17:53 2020

@author: dsg
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5)
y = np.sin(x)

plt.figure(figsize=(12,5))
plt.plot(x,y)
plt.title('x v/s y')
plt.show()

names = ['1st', '2nd', '3rd']
data = [12, 34, 54]
plt.pie(data, labels=names)
plt.show()

