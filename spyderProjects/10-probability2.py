# -*- coding: utf-8 -*-
"""
Probability lab #4

@author: Glenn
"""

import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

# 1
# You apply to 10 companies for a job, and each company offers
# jobs to 20% of (qualified) applicants.  Compute a sample of
# replies, using 0 for no offer, and 1 for offer.
x = np.random.choice(2, 10, p=[0.8, 0.2])
x = np.random.binomial(1, 0.2, size=10)

# 2 
# Repeat the experiment 10,000 times, and record in an array
# the number of offers for each experiment.
x = np.array([np.random.choice(2, 10, p=[0.8, 0.2]).sum() for i in range(10000)])
print(x)
# 3
# Plot your vector from 2 as a bar plot, showing for each possible
# value 0-10, the fraction of experiments that resulted in that value.
# Use the bar() function of matplotlib for the bar plot.  It takes
# two arguments: an array of x locations for the bars, and the values
# for the bar heights.
p = np.bincount(x)/len(x)
print(p[0])
plt.bar(list(range(11)), p)

# 4
# using your data from problem 3, create another graph, but this time,
# for value i (from 0-10) show the fraction of experiments in which
# the number of offers was less than or equal to i. The value for 10 should
# be 1.0.  (Hint: consider using function numpy.cumsum()).

# 5
# Create another plot as in part 3, but this time get your array 
# of values from  a call to function numpy.random.binomial().

# 6
# create another plot as in part 3, but this time get your array
# of values from calls to function scipy.stats.binom().  This
# function will give you the PMF of the binomial distribution, so
# you don't have to simulate a bunch of trials.

# 7
# If you have time, compute the probability of getting 2 job offers 
# after 8 applications, given the the probability of an offer in one
# application is 0.1.  Do this through simulation and through the
# use of scipy.stats.binom().


