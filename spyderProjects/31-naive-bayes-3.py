#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:42:47 2019

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

df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/heart.csv')
df['output'] = df['output'] - 1

# dummy variables
df = pd.get_dummies(df, columns=['thal', 'chestpain', 'ecg'], drop_first=True)

# compute a discrete age value
df['age_range'] = pd.cut(df['age'], bins=[0,50,60,80], labels=False)


X = df[['age', 'age_range', 'maxhr']].values
y = df['output'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

age_hd_mean = df[y == 1]["age"].mean()
age_hd_sd = df[y == 1]["age"].std()
print("Average age given heart disease {:.2f}".format(age_hd_mean))
print("Standard deviation given heart disease {:.2f}".format(age_hd_sd))

age_ok_mean = df[y == 0]["age"].mean()
age_ok_sd = df[y == 0]["age"].std()

print("Average age given no heart disease {:.2f}".format(age_ok_mean))
print("Standard deviation given no heart disease {:.2f}".format(age_ok_sd))

age = np.linspace(20, 80)
rv_hd = stats.norm(age_hd_mean, age_hd_sd)
plt.plot(age, rv_hd.pdf(age))
plt.title('density of age when no heart disease')
plt.xlabel('age')

p_hd = y_train.mean()
print("Overall heart disease probability for training data {:.2f}".format(p_hd))

p_ok = 1 - y_train.mean()
print("Overall no heart disease probability for training data {:.2f}".format(p_ok))

