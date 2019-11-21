#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:42:10 2019

@author: certinax
"""

import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/spambase-simple.csv')   

#print(df.head())

#print((df[df["spam"] == 1]).shape)

spam = df[df["spam"] == 1].shape[0]
ham = df[df["spam"] == 0].shape[0]

spam_prob = (df["spam"] == 1).mean()
ham_prob = (df["spam"] == 0).mean()
#@4
## P(w1 | ham) * P(!w2 | ham) * P(ham)

w1_ham = df[df["w1"] == 1].shape[0]/ham
Nw2_ham = df[df["w1"] == 0].shape[0]/ham

print("P(w1 | ham) * P(!w2 | ham) * P(ham) = {:.2f}".format(w1_ham*Nw2_ham*ham_prob))

#@5
## P(w1 | spam) *P(!w2 | spam) * P(spam)

w1_spam = df[df["w1"] == 1].shape[0]/spam
Nw2_spam = df[df["w1"] == 0].shape[0]/spam

print("P(w1 | spam) * P(!w2 | spam) * P(spam) = {:.2f}".format(w1_spam*Nw2_spam*ham_prob))

y = df["spam"].values
X = df.drop(columns=["spam"]).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


