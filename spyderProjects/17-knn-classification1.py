#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:49:27 2019

@author: certinax
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore


df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/College.csv', 
index_col=0)


target = np.array(df["Private"])

df.drop(columns=["Private"], inplace=True)

print(df.columns)

df = df.apply(zscore)

print(df.Accept.head(5))
print(df.Accept.min())
print(df.Accept.max())
#print(df.describe())

df = df[["Outstate","F.Undergrad"]]


x = np.random.rand(len(df))
#df.info()


