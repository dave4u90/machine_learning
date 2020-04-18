#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 23:20:00 2020

@author: dsengupta
"""

# list
x = [24, 5.0, 'Hi', True]
type(x)
len(x)
print(x[0]) # 24
print(x[3]) # True
print(x[-1]) # True
print(x[-3]) #  24
print(x[0:2]) # 24, 5.0

# tuple
# tuples, are immutable. Can process the data faster than list.

y = (24, 5.0, 'Hi', True)
type(y)
len(x)
print(y[0]) # 24
print(y[3]) # True
print(y[-1]) # True
print(y[-3]) #  24
print(y[0:2]) # 24, 5.0
# y[0] = 45 # error

# dictionary
w = { 'name': 'Debanjan', 'age': 31, 'city': 'Kolkata', 'company': 'Bekar' }
print(type(w))
print(len(w))
print(w['age'])
print(w.keys())
print(w.values())
d = [{'name': 'Debanjan', 'age': 30}, {'name': 'Utsav', 'age': 24}]
d[0]['age']

# Input from console
name = input("Enter your name ")
age = input("Enter your age  ") # input from terminal are strings
age = int(age) # typecast the string into integer
print("Hi %s, your age is %d"%(name, age))

# Control flow
if age < 18:
    print('You are underaged')
    print('You are not allowed to vote')
elif age < 18 and age < 65:
    print('You are adult')
    print('You can go to voting')
else:
    print('You are old')
    print('Eat, sleep, repeat')
    
for i in range(10):   # By default iteraror starts from 0
    print(i)
    
for i in range(3, 10): # Starring value of iterator is 3 now
    print(i)
    
for i in range(4, 15, 2): # Starting value is 4, increment by 2
    print(i)
    
temp = [21,22,24,26,43,34,5,67,21,14,10,17,56,67,39,98,22,1,39]
g20 = []
for i in temp:
    if i>20:
        g20.append(i)
        
# list comprehension
g20_new = [i for i in temp if i>20]

x = 5

while x < 20:
    print(x)
    x += 1
    
# Functions
def myfun():
    print("I am a function")
myfun()

def myfun2(a,b):
    c = a + b
    return c
myfun2(2,3)


# Packages

"""
  numpy, scipy => Mathematical computations
  pandas => Data import and export, Data cleaning, Data manipulations, Data filtering, Statistical analysis, Data exploration
  matplotlib, seaborn => Data Visualization
  sklearn (scikit-learn) => Machine learning
  theano, tensolflow => Machine learning, deep learning (Supports GPU computing)
  keras, pytorch => Advanced machine learning, advanced deep learning
  opencv, skimage => Image processing
  NLTK, Textblob, sPacy => Natural language processing
  Flask, Django => Web application
  Tkinter, PyQT => Desktop application
"""
