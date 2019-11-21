#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:17:43 2019

@author: certinax
"""

import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split
import seaborn as sns
import graphviz 

# switch to seaborn default stylistic parameters
sns.set()
# larger plot fonts
sns.set_context('talk')
# change default plot size
rcParams['figure.figsize'] = 10,8

# read the cpu data
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor']+' '+df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3/df['myct'], 2)    # clock speed in MHz



def trainz(predictors):
    target = 'prp'
    X = df[predictors].values
    y = df[target].values

    # test/train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    
    # train the model
    reg = DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, max_leaf_nodes=16)
    reg.fit(X_train, y_train)
    
    
    
    # make predictions and compute error
    y_predict = reg.predict(X_test)
    errors = y_test - y_predict
    rmse = np.sqrt((errors**2).mean())
    print('rmse: {:0.2f}'.format(rmse))
    
    return reg

# get ready for Scikit-Learn
    

predictors = ['mmin', 'chmax']      # choose predictors as you like
#trainz(predictors)

predictors = ['mmin', 'chmax', 'cach', 'cs']
#trainz(predictors)

predictors = ['cach', 'chmax', 'mmax']

tree = trainz(predictors)


# view the tree
# conda install python-graphviz
dot_data = export_graphviz(tree, precision=2,
                     feature_names=predictors,  
                     proportion=True,
                     filled=True, rounded=True)
graph = graphviz.Source(dot_data)  
graph