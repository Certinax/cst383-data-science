#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:14:11 2019

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
sns.set()
df.index = df["vendor"]+" "+df["model"]
df.drop(columns=["vendor","model"], inplace=True)
df["cs"] = np.round(1e3/df["myct"], 2)
df.info()

#sns.pairplot(df, vars=["cs", "myct", "cach", "prp"])

sns.scatterplot(data=df, x="cach", y="prp")

y = df["prp"].values
X = df[["cach"]].apply(zscore).values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

fit = LinearRegression()

fit.fit(X_train,y_train)

predict = fit.predict(X_test)

#plt.plot([X.min(), predict.min()], [X.max(), predict.max()])
print(predict.max())
plt.plot([0,250], [predict.min(),predict.max()])

sns.regplot(data=df, x="cach", y="prp")

sns.scatterplot(y_test, predict)
plt.ylabel("Predict")
plt.xlabel("Actual")
plt.title("Predicted vs actual")