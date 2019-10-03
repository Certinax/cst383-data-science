# -*- coding: utf-8 -*-
"""
Pandas dataframes

@author: Glenn Bruns
"""
import numpy as np
import pandas as pd


# allow output to span multiple output lines in the console
pd.set_option('display.max_columns', 500)

# =============================================================================
# Read data
# =============================================================================

# read 1994 census summary data
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/1994-census-summary.csv")
df.set_index('usid', inplace=True)
df.drop('fnlwgt', axis=1, inplace=True)

# =============================================================================
# Simple aggregation
# =============================================================================

# print the average age
print(np.average(df["age"]))
# get the min, max, and avg value for each numeric column
print(df.select_dtypes(include=[np.number]).aggregate(["min", "max", "mean"]))

# for a dataframe you get the aggregate for each column by default


# =============================================================================
# Aggregation with grouping
# =============================================================================

# how many people in each category of education?
# Try using pandas function value_counts().
print(df["education"].value_counts())

# for each native country, what is the average education num?
print(df.groupby("native_country").aggregate({"education_num":'mean'}))

# repeat the previous problem, sorting the output by the average
# education num value
print(df.groupby("native_country").aggregate({"education_num":"mean"}).sort_values(by="education_num"))

# for each occupation, compute the median age
print(df.groupby("occupation").aggregate({"age":"mean"}))

# repeat the previous problem, but sort the output
print(df.groupby("occupation").aggregate({"age":"mean"}).sort_values(by="age"))

# find average hours_per_week for those with age <= 40, and those with age > 40
print(df.groupby([df.age <= 40, df.age > 40]).aggregate({"hours_per_week":"mean"}))

# do the same, but for age groups < 40, 40-60, and > 60
#print(df.groupby([df.age < 40, df.age >= 40 & df.age <= 60, df.age > 60]).aggregate({"hours_per_week":"mean"}))
print(df.groupby([df.age == 60]).aggregate({"hours_per_week":"mean"}).rename(columns={'hours_per_week':'thora'}))
print(df.columns)
# get the rows of the data frame, but only for occupations
# with an average number of education_num > 10
# Hint: use filter

# =============================================================================
# Vectorized string operations
# =============================================================================

# create a Pandas series containing the values in the native_country column.
# Name this series 'country'.

# how many different values appear in the country series?

# create a Series containing the unique country names in the series.
# Name this new series 'country_names'.

# modify country_names so that underscore '_' is replaced
# by a hyphen '-' in every country name.  Use a vectorized operation.
# (See https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html)

# modify country_names to replace 'Holand' with 'Holland'.

# modify country_names so that any characters after 5 are removed.
# Again, use a vectorized operation

# Suppose we were to use only the first two letters of each country.
# Write some code to determine if any two countries would then share
# a name.

# If you still have time, write code to determine which countries
# do not have a unique name when only the first two characters are
# used.  Hint: look into Pandas' Series.duplicated().

# =============================================================================
# Handling times and dates
# =============================================================================

# read gas prices data
gas = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/Gasoline_Retail_Prices_Weekly_Average_by_Region__Beginning_2007.csv")

# create a datetime series and make it the index of the dataset

# plot the gas prices for NY city

# plot gas prices for NY city and Albany on the same plot

# if you still have time, see if you can find and plot California
# gas prices



