#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:12:23 2019

@author: certinax
"""

import numpy as np

x = np.array([4, 6, 2, 4, 9, 5])
splits = [1,4]
print(np.split(x, splits))
print(x)

y = np.array([20, 10, 30, 40])
print(y > 20)
mask = y > 20
print(y[mask])

x = np.array([4, 6, 2, 4, 9, 5])
print(x[[0,3,4,5]])
print(x[x[2:4]])
print(x[2:4])

x = np.array([range(i, i+3) for i in [2,4,6]])
print(x)
print(x[1,2])
print(x[1,])