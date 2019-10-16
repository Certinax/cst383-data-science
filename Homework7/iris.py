#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:56:21 2019

@author: certinax
"""

import numpy as np
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/iris.csv")

df.info()
#print(df.head())

# You will want to write Python code to answer these questions, but 
# I only need to see your answers.  Assume the iris data was collected 
# by randomly selecting irises.  If a number is required, give 3 significant 
# digits in your answers.  An error margin of 2% in answers is accepted.

#@1
# Use the Pandas dataframe corr() function on the iris data 
# (but omit the 'species' column).  Name the two columns that are 
# most strongly correlated, either positively or negatively.  
corrL = df.select_dtypes(include="number").corr()
np.fill_diagonal(corrL.values, np.nan)
#print(corrL)
#print(corrL.max())

##### petal_length & petal_width

#@2
# What the marginal probability that an iris is of species “setosa”?
#me = df[{"species":['setosa']}]
te = df[["species"]]
#print(te.groupby("species").size()/len(df))

### 0.333

#@3
# What is the conditional probability that an iris is of 
# species “setosa” given its sepal length is less than 5?
print(df[(df.sepal_length < 5) & (df.species.isin(["setosa"]))].shape[0]/df[df.sepal_length < 5].shape[0])

### 0.909

#@4
# What is the conditional probability that an iris has sepal 
# length less than 5 given its species is “setosa”?

print(df[df.sepal_length < 5].shape[0]/df[df["species"] == "setosa"].shape[0])

### 0.393

#@5
# What is the conditional probability that an iris has sepal 
# length greater than 6 and sepal width less than 2.6 given 
# its species is “versicolor”?

### 0.329