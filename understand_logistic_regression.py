#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:08:38 2020

@author: dsg
"""


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-50, 50) # x means age, we want y as output which will be between 0 to 1 as probability of having diabetes

# If we plot x now, this would be a linear plot

plt.plot(x)
"""
Check the y axis is not between 0 to 1, meaning linear regression is not a good fit,
because our target is not a continuous feature, rather it's a categorical feature, Yes or No.
"""
plt.show()

"""
The following function converts y into a probality curve, and is knonwn as Sigmoid fucntion
"""
y = 1/(1 + np.exp(-1*x))
plt.plot(x,y) 

# Observe y is now having value between 0 to 1
plt.show()
