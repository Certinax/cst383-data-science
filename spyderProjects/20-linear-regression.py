#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:46:55 2019

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


X = df[["cach", "chmax"]].apply(zscore).values
y = df["prp"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept: {:.3f}".format(model.intercept_))
print("Cache: {:.3f}".format(model.coef_[0]))
print("Chmax: {:.3f}".format(model.coef_[1]))

""" Model: 94 + 75*(cache) + 12*(chmin) """

R2 = model.score(X_train, y_train)

print("R^2 Score: {:.3f}".format(R2))


""" Prediction """
predicted = model.predict(X_test)

print(predicted)

