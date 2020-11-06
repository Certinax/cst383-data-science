#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:35:30 2019

@author: certinax
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from scipy import stats
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/cars-1978.csv")

df.set_index("Car", inplace=True)
print(df.info())

print(df.head())

print(df.describe())

plt.title("MPG")
plt.xlabel("Miles per gallon")
plt.ylabel("# of occurences")
plt.hist(df["MPG"].values)