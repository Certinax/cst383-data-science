#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:25:33 2019

@author: certinax
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


#@4
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/heart.csv")
df['output'] = df['output'] - 1

df = df[['age', 'maxhr', 'restbp', 'output']]
sns.scatterplot(x='age', y='maxhr', hue='output', data=df)

#@1
# 2 * p * (1 - p)

def giniVal(prob):
    return 2*prob*(1-prob)

print(giniVal(0.6))

#@2
def gini(class_counts):
    """ return the Gini value for a node in a binary classif. tree """
    # your code here     (don't forget the return statement))
    if sum(class_counts) == 0:
        return 0
    p = class_counts[0]/sum(class_counts)
    return 2*p*(1-p)

#@3
print(gini((30,50)))
print(gini((10,10)))
print(gini((20,0)))
print(gini((100,0)))


#@5
# A split on age 52 looks good.

#@6
pos = (df['output'] == 1).sum()
neg = (df['output'] == 0).sum()
print("Gini index for heart disease in the full dataset: {:.2f}".format(gini([pos,neg])))

#@7
ageSplit = 40

ageLowPos = ((df['age'] < ageSplit) & (df['output'] == 1)).sum()
ageLowNeg = ((df['age'] < ageSplit) & (df['output'] == 0)).sum()
gini_lo = gini([ageLowPos, ageLowNeg])
print("Gini index for age < 50: {:.2f}".format(gini_lo))

ageHighPos = ((df['age'] >= ageSplit) & (df['output'] == 1)).sum()
ageHighNeg = ((df['age'] >= ageSplit) & (df['output'] == 0)).sum()
gini_hi = gini([ageHighPos, ageHighNeg])
print("Gini index for age >= 50: {:.2f}".format(gini_hi))

#@8
fraction_lo = (df['age'] < ageSplit).sum()/df['age'].shape[0]
fraction_hi = (df['age'] >= ageSplit).sum()/df['age'].shape[0]
gini_split = (gini_lo * fraction_lo) + (gini_hi * fraction_hi)
print("Gini split: {:.2f}".format(gini_split))