#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:19:29 2019

@author: certinax
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor']+' '+df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3/df['myct'], 2)	# clock speed in MHz 

X = df[["cach","cs"]].apply(zscore).values
y = df["prp"].values

pf = PolynomialFeatures(degree=2, include_bias=False)

pf.fit(X)
print(pf.get_feature_names())
X_poly = pf.transform(X)
print(X[0,:])
print(X_poly[0,:])

reg = LinearRegression()
reg.fit(X_poly,y)
print("PolyScore: {:.2f}".format(reg.score(X_poly,y)))

X_train, X_test, y_train, y_test = train_test_split(X_poly,y,test_size=0.25,random_state=42)

reg.fit(X_train, y_train)

predictions = reg.predict(X_test)

rmse = np.sqrt(((predictions - y_test)**2).mean())

print("RMSE with poly: {:.2f}".format(rmse))



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

reg.fit(X_train, y_train)

predictions = reg.predict(X_test)

rmse = np.sqrt(((predictions - y_test)**2).mean())

print("RMSE no poly: {:.2f}".format(rmse))
