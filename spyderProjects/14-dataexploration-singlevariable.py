#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:52:40 2019

@author: certinax
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/1994-census-summary.csv")

##
## To display data:
##
## pd.options.display.width = 0 //// or huge #
##

#pd.options.display.width = 900

# 4 - Which Pandas commands can you use to get a quick overview of the data?
df.info()

# 5 - Remove the 'usid' and 'fnlwgt' columns from the data frame.
df.drop(columns=["usid","fnlwgt"], inplace=True)

# 6 - Use a Pandas command to look at the first rows of the data frame.
print(df.head())

# 7 - The ‘education_num’ column records the number of years of education.  
# Use ‘describe’ to find min, max, median values for education_num.  
# Plot education_num using a histogram.  Label the x axis 
# with 'years of education'.
print(df["education_num"].describe())

# 8 - Does it make sense to use education_num with a histogram?  
# Try it, and compare with a plot using a bar plot of the count of 
# the rows by education_num (as shown in lecture).
#df.education_num.hist()

# 9 - Plot capital_gain with a density plot.  
# Did you find anything interesting?  Save your plot to a png file.
#df.capital_gain.plot.density()
sns.kdeplot(df.capital_gain)
#sns.distplot(df.capital_gain, hist=False)