#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:43:56 2019

@author: certinax
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from scipy.spatial import distance_matrix
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/heart.csv")
df['output'] = df['output'] - 1	# convert to 0,1 values

df.info()

print(df.iloc[5,:])

# Categorical data:
# - chestpain
# - ecg
# - thal
# - fluor
# - exercise

print(df["exercise"].value_counts())

df = pd.get_dummies(df, columns=["chestpain", "ecg", "thal"], drop_first=True)

print(df.columns)

X = df[["sex", "sugar", "age"]].values
y = df["output"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

