#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:19:19 2019

@author: certinax
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore

df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/College.csv', 
index_col=0)

target = df["Private"].values

df.drop(columns="Private", inplace=True)

#print(df.columns.values)

df = df.apply(zscore)

#print(df["Apps"]) // Scaling - OK

df = df[["Outstate", "F.Undergrad"]]

x = np.random.rand(len(df)) # This code generates a random number between 0 and 1 for the length of df

mask = np.random.rand(len(df)) < 0.75 # This codes gives us random number for 75% of data set, this will be used as traning set

""" Creating training and test data """
tr_dt = df[mask]
te_dt = df[~mask]

""" Labels """
tr_labels = target[mask]
te_labels = target[~mask]

def edist(x,y):
    return np.sqrt(np.sum((x-y)**2))

x = tr_dt.values
x1 = df[df.index == "Chestnut Hill College"].values
print(x1)
k = 3
dists = np.apply_along_axis(lambda x: edist(x, x1), 1, x)
topk = np.argsort(dists)[:k]
print(target[topk])

print(target[df.index == "Chestnut Hill College"])
