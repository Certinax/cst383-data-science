#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:56:20 2019

@author: certinax
"""

import numpy as np

y_test = np.array([3.1,1.5,2.6])
y_predict = np.array([3.0,2.2,2.4])

print(((y_test-y_predict)**2).mean())

print(np.array([1,2,3,3]))
