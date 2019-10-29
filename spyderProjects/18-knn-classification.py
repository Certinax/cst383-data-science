#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:42:55 2019

@author: certinax
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/iris.csv', index_col=4)

df = df.apply(zscore)

X = df.values
y = df.index.values


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

#accuracy = knn.score(X_test, y_test)

#print(accuracy)

#blind_prediction = pd.Series(y_train).value_counts()[0]

#print(blind_prediction)

#baseline_accuracy = 
