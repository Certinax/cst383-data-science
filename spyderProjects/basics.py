#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:32:08 2019

@author: certinax
"""

import numpy as np

x = np.array([6.4, 8.2, 8.8, 9.2, 7.2])

sum = 0

for i in x:
    sum += i
    
avg = sum/len(x)

print(round(avg, 2))

print((sum(x)/len(x)))