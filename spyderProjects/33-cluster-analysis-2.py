#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:40:15 2019

@author: certinax
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from scipy import stats
import seaborn as sns
from scipy.stats import zscore

file = 'https://raw.githubusercontent.com/grbruns/cst383/master/winequality-red.csv'
df = pd.read_csv(file, sep=';')
# don't use quality
df.drop('quality', axis=1, inplace=True)
# scale the data
df = df.apply(zscore)


X = df.iloc[:,:5].values

# plot cluster centers
k = 3
kmeans = KMeans(n_clusters=k)

kmeans.fit(X)
centers = kmeans.cluster_centers_
counts = np.bincount(kmeans.labels_)

fix, ax = plt.subplots(k, 1, figsize=(10,8))
for i, axi in enumerate(ax.flat):
	axi.bar(range(n), centers[i,:])
	axi.set_title('number in cluster: {}'.format(counts[i]))
	axi.set_ylim(-3.5, 3.5)
	axi.set_xticks(range(n))
	axi.set_xticklabels(df.columns[:n])
plt.tight_layout()

