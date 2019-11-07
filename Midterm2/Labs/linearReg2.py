#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:02:55 2019

@author: certinax
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore


df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor']+' '+df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3/df['myct'], 2)	# clock speed in MHz 

def zz(x):
 return (x - x.mean())/x.std()

X = df[["cach","cs"]].apply(zz).values
y = df["prp"].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

lin = LinearRegression()
lin.fit(X_train, y_train)

print("Intercept: {:.2f}".format(lin.intercept_))
print("Cache: {:.2f}".format(lin.coef_[0]))
print("CS: {:.2f}".format(lin.coef_[1]))

mse1 = ((lin.predict(X_train) - y_train)**2).mean()
mse2 = ((y_train - y_train.mean())**2).mean()

r2 = (mse2-mse1)/mse2

print("R^2: {:.2f}".format(r2))

sns.scatterplot(lin.predict(X_train), y_train)
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.title("Predicted vs actual")