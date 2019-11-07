#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:39:15 2019

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

print(df.describe())

#df = df.apply(zscore)

#sns.distplot(df["chol"])
sns.pairplot(df, vars=['age', 'exercise', 'chestpain',
'chol'], markers='.', height=2)

corr = df.corr()

#sns.heatmap(corr)

#sns.scatterplot(data=df, x="age", y="fluor")