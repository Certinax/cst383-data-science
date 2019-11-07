#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:44:38 2019

@author: certinax
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/iris.csv', index_col=4)

df = df.apply(zscore)

print(df.columns)

X = df.values
y = df.index


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#predictions = knn.predict(X_test, y_test)

accuracy = knn.score(X_test, y_test)

print(accuracy)

blind_prediction = pd.Series(y_train).value_counts().index[0]
print(blind_prediction)

baseline_accuracy = np.mean(y_test == blind_prediction)

print(baseline_accuracy)

""" REGRESSION """

df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/iris.csv', index_col=4)

target = df["petal_width"]
y = target.values

X = df.drop(columns="petal_width")

X = X.apply(zscore).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(X_train, y_train)

predicted = reg.predict(X_test)

mse = ((predicted - y_test)**2).mean()

print("MSE: {:.2f}".format(mse))

base_mse = ((y_train - y_test.mean())**2).mean()

print("MSE for blind prediction: {:.2f}".format(base_mse))

rmse = np.sqrt(((predicted - y_test)**2).mean())

print("RMSE: {:.2f}".format(rmse))

mse1 = ((y_train - reg.predict(X_train))**2).mean()
mse2 = ((y_train - y_test.mean())**2).mean()

r2 = (mse2-mse1)/mse2
print("R^2: {:.2}".format(r2))

