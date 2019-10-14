#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:46:56 2019

@author: certinax
"""

import numpy as np
import pandas as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/College.csv")

df.info()

sns.scatterplot(x='F.Undergrad', y='Expend', color="darkgreen", marker="o", s=35, data=df)