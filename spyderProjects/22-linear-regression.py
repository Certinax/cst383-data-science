#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:40:03 2019

@author: certinax
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from scipy.spatial import distance_matrix
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor']+' '+df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3/df['myct'], 2)	# clock speed in MHz 

print(df.columns)

y = df["prp"].values
df.drop("prp", inplace=True, axis=1)
df = df.apply(zscore).values
X = df[:,[1,2,4,5]]


def generateResults(testSize):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)
    
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    y_predict = reg.predict(X_test)
    
    mse = ((y_predict - y_test)**2).mean()
    #print(mse)
    
    rmse = np.sqrt(mse)
    #print(rmse)
    print("Testsize {:.2f}, MSE: {:.2f}, RMSE: {:.2f}".format(testSize, mse, rmse))

testSizes = np.random.random_sample((4,))
randState = np.random.randint(1,101)
for i in testSizes:
    generateResults(i)

#score = reg.score(X_test, y_test)
#print(score)
