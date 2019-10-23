#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:44:18 2019

@author: certinax
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/CruddyShad0w/CST-383-CrashData/master/Crash_Reporting_-_Drivers_Data.csv"  
  
df = pd.read_csv(url)


# Setting the time and date to proper pandas time /date
df.set_index(pd.to_datetime(df['Crash Date/Time']), inplace=True)
df.sort_index(ascending=False, inplace=True)
df.head()
df.info()
df.tail(1)
df.isna().mean()
df.isna().mean(axis=1)

df.isna()
sns.scatterplot

data = pd.DataFrame(df["Report Number"].value_counts().value_counts().sort_index())
data
sns.barplot(x=data.index, y="Report Number", data=data)

df["Report Number"].unique().shape
print(df.index.min())
print(df.index.max())

timeframe = df.index.max() - df.index.min()
type(timeframe)

startdate = df.index.min()
enddate = df.index.max()


sns.FacetGrid()


#df["Driver At Fault"].value_counts()
#sns.countplot(df["Driver At Fault"].value_counts())
#print(df.index)
#print(len(df.columns))
#fault = pd.get_dummies(df['Driver At Fault'], drop_first=True)
#print(fault.columns)

#df["Numeric Fault"] = df['Driver At Fault'].replace(('Yes','No'), ('1','0'), inplace = True)

#print(df["Vehicle Year"].value_counts())

#print(df["Speed Limit"].value_counts())

#print(df["Drivers License State"].value_counts())