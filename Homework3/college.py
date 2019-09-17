# -*- coding: utf-8 -*-
"""

CST 383, Homework 3.

In this homework we use Pandas to investigate a data set about 
college majors.  The data is from 2010-2012, and was derived
from US Census data by an Ben Casselman of fivethirtyeight.com

"""

"""

Info about the data set.

Here is the data:
https://github.com/fivethirtyeight/data/tree/master/college-majors

Here is Casselman's report:
https://fivethirtyeight.com/features/the-economic-guide-to-picking-a-college-major/

This data set concerns people under 28 who have
a bachelor's degree but no graduate degree.  The data comes
from the US Census PUMS data, which is a statistical sample of the 
US census data.  

Information about the columns in the data set can be found on the
github page listed above.  It's hard to interpret some of this data.  
For example, unemployed + employed does not equal total.  From the R 
script Casselman wrote to generate his data set, this is what I understand:
    
Total: total number of people for the major (in the census data)
  
Median: median yearly income for those in the major

Full_time_year_round: 50+ weeks/year, 35+ hours/week.
  
Employed:
    - employed means employment status code is either
       - civilian employed, at work
       - civilian employed, with a job but not at work
         (this means the person does not happen to be at work that day, for
          reasons including vacation, illness, bad weather, strike, etc.)
    - note that this can be part-time work, or even erratic work, I guess
    
Unemployed: census employment status code is 'unemployed'
    - Employed + Unemployed does give give the total for the major, because
      there are other categories, like 'not in labor force', and 'armed forces'

Full_time: avg hours/week > 35 in last 12 months 

Part-time: average hours/week over past 12 months was non-zero, but less than 35 hours.
   - Note that Full_time + Part_time gives total number of people who
     worked over the past 12 months, but does not give the same value
     as Employed.
         
College jobs, Non_college_jobs, Low_wage_jobs:  These are related to
  certain lists of jobs, and if you add the values together, you don't
  get all job.  Casselman doesn't show exactly which jobs these are.

"""

"""

Homework instructions:
    
PLEASE NOTE: your code will be graded automatically so please
follow these instructions carefully.  My grading script grades
each of your answers *independently*.  In other words, I do not
run all of your submitted code in order.  I take your code for
one problem, and run it in an environment where certain variables
or functions are defined.  For each problem I will list the 
variables you can assume to be defined in your answer.  Do not
assume that variables or functions that you defined earlier are
available.

In your code for a problem, you can define temporary methods and
variables before your final line of code, but you cannot assume
these temporary methods are available in later problems.

Also, please note that you should not use a loop anywhere in your code
for this homework.

"""

# note: do not import other packages; my test script uses only these
import numpy as np
import pandas as pd

# allow output to span multiple output lines in the console
pd.set_option('display.max_columns', 500)

#
# read the data
#

df = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/college-majors/recent-grads.csv")

#
# series
#

# let's start with two series
# (without the copy, we are just getting a view of a column,
#  and it will cause problems later, for example in sorting)
unemp = df['Unemployment_rate'].copy()
major = df['Major'].copy()

#@ 1
# Compute the length of the 'major' series.  Your result should
# be a single integer.
# assume major is defined
# (write an expression)
major.size


#@ 2
# Compute the first 10 elements of the data in the unemp series.
# assume unemp is defined
# (write an expression)
unemp.iloc[:10]

#@ 3
# Compute the maximum value of the unemp series (use a pandas method).
# assume unemp is defined
# (write an expression)
unemp.max()

#@ 4
# Set the index of unemp to be the data in the major series.
# assume unemp and major are defined
# (modify unemp)
unemp.index = [major]

#@ 5
# Compute the 10 majors with the lowest unemployment rate, sorted 
# by increasing unemployment rate.
# The first lines of your output should look like this:
# EDUCATIONAL ADMINISTRATION AND SUPERVISION    0.000000
# MILITARY TECHNOLOGIES                         0.000000
# BOTANY                                        0.000000
# MATHEMATICS AND COMPUTER SCIENCE              0.000000
# SOIL SCIENCE                                  0.000000
# ENGINEERING MECHANICS PHYSICS AND SCIENCE     0.006334
# assume unemp is defined, and is indexed by majors
# (write an expression)
unemp.sort_values()[:10]

#@ 6
# Compute the number of majors that have unemployment levels less than 0.05
# assume unemp is defined
# (write an expression)
unemp[unemp < 0.05].size

#@ 7
# Compute the fraction of majors that have unemployment levels less than 0.06
# (note: 0.06, not 0.05)
# Your result will be a value between 0 and 1.
# assume unemp is defined
# (write an expression)
unemp[unemp < 0.06].size / unemp.size

#@ 8
# Compute the fraction of majors that have unemployment levels lower than the
# unemployment level of 'COMPUTER SCIENCE'.
# assume unemp is defined
# (write an expression)
#print(unemp["COMPUTER SCIENCE"].values)
#print(unemp["COMPUTER SCIENCE"])
#print(unemp["COMPUTER SCIENCE"].item())
unemp[unemp < unemp["COMPUTER SCIENCE"].item()].size / unemp.size

#
# data frames
#

#
# preprocessing
#

# make the major the index, and remove the major column
# also, drop Major_code and Sample_size
df.index = df['Major']
df.drop('Major', axis=1, inplace=True)
df.drop(['Major_code', 'Sample_size'], axis=1, inplace=True)

# any missing data?  (We haven't covered this yet.)
df.isna().sum(axis=1).sort_values(ascending=False)[:10]

# drop rows with NA
df.dropna(inplace=True)


###############################################################
# In your answers to all remaining problems you can assume
# that df is defined, and has the columns of the dataframe df
# just defined.
###############################################################


#@ 9
# Compute a series containing the values for major 'MICROBIOLOGY'.
# (write an expression)
df.loc["MICROBIOLOGY"]

#@ 10
# Compute the total number of people represented in the
# data set (in other words, sum the values in the 'Total' column)
# (write an expression)
df["Total"].sum()

#@ 11
# Compute the overall fraction of women
# Your result should be a number between 0.55 and 0.60
# (write an expression)
df["Women"].sum() / df["Total"].sum()

#@ 12
# Compute the major with the highest value of ShareWomen
# Your result should be a single string value.
# (write an expression)
df[df["ShareWomen"] == df["ShareWomen"].max()].index[0]

#@ 13
# Compute the median earnings for all the majors with
# share of women > 90%.  The first rows of your result
# should look like this:
# Major
# MEDICAL ASSISTING SERVICES                       42000
# SPECIAL NEEDS EDUCATION                          35000
# ELEMENTARY EDUCATION                             32000
# (write an expression)
df[df["ShareWomen"] > 0.9]["Median"]

#@ 14
# Compute a series containing the median earnings for all the 
# majors with share of women < 15%.  The first rows of your
# result should look like this:
# Major
# PETROLEUM ENGINEERING                          110000
# MINING AND MINERAL ENGINEERING                  75000
# NAVAL ARCHITECTURE AND MARINE ENGINEERING       70000
# (write an expression)
df[df["ShareWomen"] < 0.15]["Median"]

#@ 15
# Compute the ratio of the median of the highest earning major
# to the median of the lowest earning major.  Your output should
# be a single number, and it should be close to 5.
# (write an expression)
df["Median"].max()/df["Median"].min()

#@ 16
# Compute the top 10 majors by median earnings.  For each of these
# majors, your result should contain the median, .25 percentile, and
# .75 percentile earnings.  Sort the result by median earnings, largest first.
# The first lines of your output should look like this:
#                                            Median  P25th   P75th
# Major                                                           
# PETROLEUM ENGINEERING                      110000  95000  125000
# MINING AND MINERAL ENGINEERING              75000  55000   90000
# METALLURGICAL ENGINEERING                   73000  50000  105000
# NAVAL ARCHITECTURE AND MARINE ENGINEERING   70000  43000   80000
# (write an expression)
df[["Median", "P25th", "P75th"]].sort_values(by="Median", ascending=False)[:10]

#@ 17
# Repeat the previous problem, but compute the 10 majors with the
# lowest median earnings, and sort with lowest meadian salary first
# (write an expression)
df[["Median", "P25th", "P75th"]].sort_values(by="Median", ascending=True)[:10]


#@ 18
# For each major, compute the fraction of people who have a non-college
# job (in other words, a job not requiring a college degree).
# Your result should contain only the top 10 majors, sorted in decreasing
# order by fraction of people.  The first lines of your output
# should look like this:
# Major
# COSMETOLOGY SERVICES AND CULINARY ARTS                               0.702569
# NUCLEAR, INDUSTRIAL RADIOLOGY, AND BIOLOGICAL TECHNOLOGIES           0.697070
# ELECTRICAL, MECHANICAL, AND PRECISION TECHNOLOGIES AND PRODUCTION    0.681314
# (write an expression)
(df["Non_college_jobs"]/df["Total"]).sort_values(ascending=False)[:10]


#@ 19
# For each major category, compute the total number of people
# in that category.  Your result should be sorted by number of people, descending order.
# You will need the 'Total' column.
# The first lines of your output should look like this:
# Major_category
# Business                               1302376.0
# Humanities & Liberal Arts               713468.0
# (write an expression)
(df.groupby("Major_category").aggregate({'Total':'sum'})).sort_values(by="Total", ascending=False)

#@ 20
# For each major category, compute the fraction of people
# associated with that category.  Order your output by fraction
# of people, in decreasing order.
# The first lines of your output should look like this:
# Major_category
# Business                               0.192328
# Humanities & Liberal Arts              0.105361
# Education                              0.082569
# (write an expression)
((df.groupby("Major_category").aggregate({'Total':'sum'}))/df["Total"].sum()).sort_values(by="Total", ascending=False)


#@ 21
# Compute the number of majors associated with each major
# category.  The first lines of your result should look like this:
# Engineering                            29
# Education                              16
# Humanities & Liberal Arts              15
# Biology & Life Science                 14
#
# (write an expression)
df.groupby(["Major_category"])["Total"].count().sort_values(ascending=False)

#@ 22
# Add a new variable 'HighShareWomen' to the data frame.
# This variable should be True if ShareWomen > 0.50, and
# False otherwise.
# (modify df)


#@ 23
# Compute the fraction of majors that have a high share of women.
# Your result should be a number close to 0.56
# (write an expression)


#@ 24
# For each major, compute the fraction of those working in
# low-end jobs.  Show the 10 majors with the most low wage
# jobs, ordered by decreasing fraction of low-end jobs.
# The first lines of your output should look like this:
# Major
# COSMETOLOGY SERVICES AND CULINARY ARTS    0.300951
# DRAMA AND THEATER ARTS                    0.255913
# MISCELLANEOUS FINE ARTS                   0.226048
# CLINICAL PSYCHOLOGY                       0.219168
# STUDIO ARTS                               0.211227
# (write an expression)


#@ 25
# For each major category, compute the maximum of the median
# salaries for majors within that category.  Sort by decreasing
# median salary.
# The first lines of your output should look like this:
# Major_category
# Engineering                            110000
# Physical Sciences                       62000
# Business                                62000
# (write an expression)


#@ 26
# Compute the average median salary for each major category
# sort by increasing salary
# The first lines of your output should look like this:
# Major_category
# Psychology & Social Work               30100.000000
# Humanities & Liberal Arts              31913.333333
# Education                              32350.000000
# (write an expression)


# The answer to the previous question does not accurately
# represent the median salary for a major category, because
# certain majors within a category may have a lot more students
# than other categories.  When combining the salaries of majors
# within a major category, we need to take into account the
# number of people in each major within the category. If a
# major has lots of people, the median salary for that major
# should have a bigger weight in get the median salary for
# the major category.

#@ 27
# Create a new column 'Major_share' that gives, for each major, the fraction of people 
# associated with the major. The sum of the new column should be 1.0
# or very close to 1.0.
# (write a statement)    (note: formerly this incorrectly asked for an expression)


###############################################################
# In the remaining problems you can assume that df additionally
# contains the column 'Major_share'
###############################################################


#@ 28
# Now compute the weighted median salary for each major category,
# taking into account the number of people in each major
# You will probably want to use 'groupby' in conjunction with
# apply().  Don't forget the variable 'Major_share'.  You
# may want to define the function that gets applied.
# Your first lines of output should look like this:
# Major_category
# Agriculture & Natural Resources        34951.719122
# Arts                                   31716.044578
# Biology & Life Science                 34446.722572
# Business                               40942.111188
# (write an expression)


#@ 29
# Using the same idea as the last problem to compute
# the mean unemployment rate for each major category.
# Sort your result by median unemployment rate.
# Your first lines of output should look like this:
# Major_category
# Agriculture & Natural Resources        0.051505
# Arts                                   0.089105
# Biology & Life Science                 0.070219
# (write an expression)


#@ 30
# Compute the share of women by major category.  to do this
# you need to find the total number of women and total number
# of people in each category.
# The first lines of your output should look like this:
# Major_category
# Agriculture & Natural Resources        0.466318
# Arts                                   0.623694
# Biology & Life Science                 0.592566
# Business                               0.487205
# (write an expression)


#@ 31
# compute the mean median salary for majors with
# a high share of women and for majors without a high
# share of women.  You need to consider
# the number of students in each major to do this,
# because some majors have many fewer students than
# others.  Consider using the 'Major_share' column.
# Your output should look like this:
# High_ShareWomen
# False    43633.592490
# True     34692.752128
# dtype: float64
# (write an expression)


#@ 32
# Compute the fraction of low wage jobs by major category.
# Order your result from low to high values.
# The first lines of your output should look like this:
# Major_category
# Engineering                            0.046651
# Computers & Mathematics                0.053965
# Health                                 0.067504
# (write an expression)


#@ 33
# Compute the unemployment rate by major category.
# Your result should be sorted by unemployment rate, descending.
# The first lines of your result should look like this:
# Major_category
# Social Science                         0.096689
# Arts                                   0.089233
# Humanities & Liberal Arts              0.085852
# Law & Public Policy                    0.085258
# (write an expression)







