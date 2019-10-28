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
from scipy.stats import zscore

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

df["Collision Type"].value_counts()
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


df["Try"] = df["Weather"].value_counts()/df.shape[0]

df["zyear"] = df[["Vehicle Year"]].apply(zscore)

df["zyear"]

df["Driver At Fault"].value_counts()

df[(df["Vehicle Year"] > 1850) & (df["Vehicle Year"] < 2020)]["Vehicle Year"].value_counts().sort_index()


print(df.groupby(index).shape)

df["ACRS Report Type"].value_counts()

f = (df.groupby(["Driver At Fault", "ACRS Report Type"]).size())


dfNo = df[(df["Driver At Fault"] == "No") & (df["ACRS Report Type"] == "Fatal Crash")][["Driver At Fault", "ACRS Report Type"]].
.plot(kind="line")
dfYes = df[df["Driver At Fault"] == "Yes"].shape[0]
dfUn = df[df["Driver At Fault"] == "Unknown"].shape[0]

df.info()


#Displaying Types of injuries based on the speed limit and if the driver was responsible for the accident
sns.boxplot(x="ACRS Report Type",y="Speed Limit",hue="Driver At Fault",data=df, palette="coolwarm")


#Displaying the data for an overview
sns.countplot(x="ACRS Report Type", data=df)

#Displaying the people injuries
tbl = pd.crosstab(df['Vehicle Body Type'], df['ACRS Report Type'])
tbl = tbl.div(df['Vehicle Body Type'].value_counts(), axis=0)
tbl.plot(kind='bar')

sns.catplot(x='ACRS Report Type', y= 'Speed Limit', hue = 'Driver At Fault', kind='swarm', data = df)

print(df["ACRS Report Type"].isna().sum())
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


rng = pd.to_datetime([1349720105, 1349806505, 1449892905, 1349979305, 1350065705], unit='s')

ts = pd.Series(np.random.randn(len(rng)), index=rng)

ts["2012":"2015"]