#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 07:32:20 2020

@author: dsg
"""


import numpy

x = [1, 5, 8]
y = [3, 5, 6]

# The following will concat the two lists, because they are treated as data entities and are not treated as mathematical entities
x + y

a = numpy.array([10, 20, 30])
b = numpy.array([15, 25, 35])

# The following will add/subtract etc. with respective array numbers since they are now treated as mathematical entities.
# List is a data entity, whereas array is a mathematical entity

a + b
a - b
a * b
a / b

numpy.log(x) # Logarithm
numpy.sin(x) # Sin
numpy.exp(x) # Exponential

# Creating array using range
x = numpy.arange(10) # 0 to 9
x = numpy.arange(5, 10) # 5 to 9
x = numpy.arange(4, 10, 2) # 4, 6, 8

y = numpy.linspace(0, 10, 5) # 5 values between 0 to 10 equally spaced

z = numpy.zeros(10) # 10 zeroes
z = numpy.ones(10) # 10 1s
w = numpy.ones((4,3)) # 4 rows and three columns

# Random
r = numpy.random.random()
f = numpy.random.random((4,2)) # 4 rows and 2 columns
i = numpy.random.randint(0, 50, 5) # 0 to 50, 5 random integer
i = numpy.random.randint(0, 50, (2,6)) # 2 rows and 6 columns

# Analysis
i.min() # Min value in i
i.max() # Max value in i
i.mean() # Mean
i.var() # Variance

i.min(axis=1) # Array of rowwise mins
i.max(axis=0) # Array of columnwise maxs
# Note: Axis is applicable to all the analysis functions above


# Solve linear equations
# 2x + 3y = 18
# 4x - 2y = 4

a = [[2,3], [4, -2]]
b = [18, 4]
numpy.linalg.solve(a,b)

# Matrix operations
m = [[2,3,5],[5,6,2],[4,9,8]]
numpy.linalg.inv(m) # Inverse
numpy.linalg.det(m) # Determenant
