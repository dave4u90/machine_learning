#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Predict the price of a house using Gradient Descent Algorithm.
  We will be using the 'theano' and 'numpy' package. We wil take the data as simple arrays and will try to scale it before we pass this
  to the algorithm.
"""

import theano
import numpy

x = theano.tensor.fvector('x')
y = theano.tensor.fvector('y')

# Intial staring value of m and c were 0.5 and 0.9. After training the algorithm we got the new values.
# If we put these following values, the last two cost function values will be
# 0.11963058480429081
# 0.11724993616734594
# Which shows very close to 0 values, also the change in cost function is minimal.
# We had to train the algorithm only by 100 iterarions which is a good optimisation and balances the learning rate and the iterations.
# This depends on the threshold which is required for business. For this example, the threshold is limited to two decimal places.
m = theano.shared(0.7, 'm')
c = theano.shared(-1.25, 'c')

yhat = m * x + c

# Cost function
cost = theano.tensor.mean(theano.tensor.sqr(y-yhat))/2

# Learning Rate, initially started with 0.1
lr = 0.01

# Get the partial differential of m and c
gradm = theano.tensor.grad(cost,m)
gradc = theano.tensor.grad(cost,c)

# Get new value of m and c
mn = m - lr * gradm
cn = c - lr * gradc

# Create a function that will update the m and c with mn and cn respectively
train = theano.function([x,y], cost, updates = [(m, mn), (c, cn)])

# Now we will take x(area) and y(price) as an array

area = [120, 180, 215, 350, 450, 195, 256, 452, 367, 874, 652]
price = [200, 250, 390, 410, 540, 210, 345, 545, 451, 754, 658]

# NumPy arrays are 64 bit, theano works with 32 bit arrays. This is why we need this following typecast.
area = numpy.array(area).astype('float32') / numpy.std(area)
price = numpy.array(price).astype('float32') / numpy.std(price)

# Scale the data
area = area - numpy.mean(area)
price = area - numpy.mean(price)

# Train the algorithm by a number of iterations
for i in range(100):
    cost_val = train(area, price)
    print(cost_val)
    
print(m.get_value())
print(c.get_value())

# Function to predict the price of a house when given the area
prediction = theano.function([x], yhat)
prediction([345.0])    