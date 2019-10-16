#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:24:22 2019

@author: certinax
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/College.csv")

df.info()

#3
# Derive a new column, ‘Size’, from the F.Undergrad variable.  
# The possible values of Size should be “small”, “medium”, or “large”.   
# The value “small” should be assigned to the colleges in the “bottom 3rd” 
# of F.Undergrad values, “medium” should be assigned to the “middle 3rd”, 
# and “large” to the “top 3rd”.  Use the Pandas ‘quantile’ function to 
# find the corresponding F.Undergrad values.  (If you're not sure how to 
# do this, see the hints right away).
#breaks = df["F.Undergrad"].quantile([0.33, 0.66, 1.0])
#df["Size"] = pd.cut(df["F.Undergrad"], include_lowest=True, bins=breaks, labels=['small','medium','large'])

breaks = df['F.Undergrad'].quantile([0,0.33, 0.66, 1.0])
df['Size'] = pd.cut(df['F.Undergrad'],
   			include_lowest=True, bins=breaks,
   			labels=['small', 'medium', 'large'])

#4
#g = sns.FacetGrid(df, col='Size', height=4, aspect=0.8)
#g.map(plt.scatter, 'PhD', 'Outstate', s=20, color="green")

#5
#sns.scatterplot(x="PhD", y="Outstate", hue="Size", data=df)

#6
#sns.scatterplot(x="PhD", y="Outstate", hue="Size", size="Size", data=df)

#7
#sns.violinplot(y="Outstate", x="Size", data=df)

#8
sns.violinplot(y="Outstate", x="Size", data=df, inner="stick")
