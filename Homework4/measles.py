# -*- coding: utf-8 -*-
"""

CST 383, measles simulation homework

# Here's a question.  Suppose 1% of people have measles, that the
# test for measles if 98% accurate if you do have measles, and 98%
# accurate if you don't have measles.  Then what is the probability
# that you have measles, given that you have tested positive for them?
#
# Try guessing an answer before you start on this assignment.
#
# In this homework we will use simulation to estimate the answer,
# and we'll also compute the answer using Bayes' Law.  There
# are three parts below:
# 1. Warm up by simulating some coin flips.
# 2. Use simulation to answer the question above.
# 3. Use Bayes' Law to answer the question without simulation.

"""

import numpy as np
import matplotlib.pyplot as plt


# Instructions: 
# Problems start with #@ and then give a number.  Enter your
# Python code after each problem.  Do not use any variables
# in your answer except for the ones that the problem says
# you can assume are defined.


#
# Part 1: warmup
#

#@ 1
# Simulate flipping a coin 200 times that has a 90% chance of
# landing heads.  Store your result in a NumPy array x of length
# 200 that contains only 0 or 1, where 1 represents heads.
# Use np.random.choice().  
# (assignment to x)
x = np.random.choice(2, 200, p=[0.1, 0.9])

#@ 2
# Repeat the problem above, but this time use np.random.sample(),
# which gives values between 0 and 1.  Obviously you will need to do
# further processing to turn the output of sample() into your
# array x.  This will take a little thought.
# (assignment to x)
x = np.random.sample(size=200)

weights = [0.1, 0.9]
cs = np.cumsum(weights)

def calculated_weights(x):
    return sum(cs < x)

vectroized_calculated_weights = np.vectorize(calculated_weights)
x = vectroized_calculated_weights(x)

#@ 3
# compute the fraction of values in array x that are 1.
# (expression)
len(x[x == 1]) / len(x)


#@ 4
# Flip the weighted coin of problem 1 200 times, compute the fraction
# of values that are 1, and repeat this entire process 100 times to
# get an array of length 100.  Assign this array to variable y1.
# (assignment to y1)
def t200():
    return np.random.choice(2, 200, p=[0.1, 0.9])

y1 = np.array([len(t200()[t200()==1])/len(t200()) for i in range(100)])

#@ 5
# plot a histogram of y1 using matplotlib
# (produce a plot)
#plt.hist(y1)
#plt.title("Fraction of 1's for 200 biased coin tosses a 100 times")
#plt.xlabel("Fraction of 1's in a given attempt (of 200 tosses)")
#plt.ylabel("frequency")

#@ 6
# compute a NumPy array y2 that is just like y1, except that in creating y2
# we do 1000 coin flips in each experiment, not 200.
# (assignment to y2)
def t1000():
    return np.random.choice(2, 1000, p=[0.1, 0.9])

y2 = np.array([len(t1000()[t1000()==1])/len(t1000()) for i in range(100)])


#@ 7
# plot histograms for y1 and y2, with the histogram for y1 above 
# the plot for y2.  Our lecture notes show how to do this; see
# the 'multiple subplots' slide.  Use matplotlib.  In both histograms, 
# let the x axis values range from 0.85 to 0.95.  Please study
# the two histograms and think about why they are different.
# Assume y1 and y2 are defined.
# (produce a plot)

fig, ax = plt.subplots(2)
fig.suptitle("Histograms for Y1 and Y2")
ax[0].hist(y1)
ax[1].hist(y2)
ax[0].set_xlim([0.85, 0.95])
ax[1].set_xlim([0.85, 0.95])

#
# Part 2 - simulate the answer to the question
#

#@ 8
# Simulate the overall occurrence of measles among 10,000 people,
# based on the assumption that each person has a 0.01% chance of
# having measles.  
# Compute a NumPy array x of length 10,000, where each value is 
# either 0 or 1.  Each of the 10,000 values should be found by 
# "flipping a 0/1 coin" that is weighted 99% to 0.  Approximately 
# 99% of the values in x should be 0, and the others should be one.
# (assignment to x)
x = np.random.choice(2, 10000, p=[0.9, 0.1])


#@ 9
# Simulate the measles test results on the people without measles,
# based on the assumption that the measles test gives the right
# answer about 95% of the time on people without measles.
# Create an array y0, which is as long as the number of 0's in
# array x, by flipping a 0/1 coin that is weighted 95% to 0.
# Assume x is defined.
# (assignment to y0)


#@ 10
# Simulate the measles test results on the people with measles,
# based on the assumption that the measles test gives the right
# answer about 98% of the time on people with measles.
# Create an array y1, which is as long as the number of 1's in
# array x, by flipping a 0/1 coin that is weighted 98% to 1.
# Assume x is defined.
# (assignment to y1)


#@ 11
# Collect the measles-free people among those who tested positive.
# Compute a vector pos_no_meas that is all 0's, and is as long as the
# number of 1's in y0.
# Assume y0 is defined.
# (assignment to pos_no_meas)


#@ 12
# Collect the measles-infected people among those who tested positive.
# Compute a vector pos_with_meas that is all 1's, and is as long as
# the number of 1's in y1.
# Assume y1 is defined.
# (assignment to pos_with_meas)


#@ 13
# Collect information about all people who tested positive.
# Concatenate arrays pos_no_meas and pos_with_meas, and assign
# the result to array 'tested_pos'.  A 0 in in this array means 
# no measles; a 1 means measles.
# Assume pos_no_meas and pos_with_meas are defined.
# (assignment to tested_pos)


#@ 14
# Estimate the probability of having measles if you've tested
# positive for measles.  Compute the fraction of values in 
# tested_positive that are 1, and assign the result to 
# variable 'p'.
# Assume tested_pos is defined.
# (assignment to p) 


#@ 15
# Package up your code into a function 'prob_cond_given_pos'.  This
# function will return the probability of having a condition, based
# on certain probabilities.
# The function should have the following parameters:
#   prob_cond              - probability of a condition (above you used 0.01)
#   prob_pos_given_cond    - probability of testing positive given condition (you used 0.98)
#   prob_neg_given_no_cond - probability of testing negative given no condition (you used 0.95)
# The function must return the probability of having the condition.
#
# Your function should return a slightly different value every time.
# When you run prob_cond_given_pos(0.01, 0.98, 0.95), you should get an answer
# similar to the value of p you just computed.
#
# Here is the output from tests I ran with my code:
# test 1:
#     np.array([prob_cond_given_pos(0.5, 0.9, 0.8) for i in range(1000)]).mean()
#     output: 0.8180582615720287
# test 2:
#     np.array([prob_cond_given_pos(0.3, 0.8, 0.7) for i in range(1000)]).mean()
#     output: 0.5334712339397902
# test 3:
#     np.array([prob_cond_given_pos(0.5, 0.9, 0.8) for i in range(100)]).std()
#     output: 0.00550051982001144
#
## I provided the function header.  You should fill out the function body,
# including the return statement.
# (define a function)

#def prob_cond_given_pos(prob_cond, prob_pos_given_cond, prob_neg_given_no_cond):
    # YOUR CODE HERE

#
# Part 3 - compute the answer using Bayes' Law
#

#@ 16
# Write a function 'prob_cond_given_pos_bayes'.  This function
# will take the same parameters as prob_cond_given_pos, but will
# use Bayes' Law to compute the result.
#
# Here is some output from my code:
# test1:
#    prob_cond_given_pos_bayes(0.5, 0.9, 0.8)
#    output: 0.1818...
# test 2:
#    prob_cond_given_pos_bayes(0.3, 0.8, 0.7) 
#    output: 0.5333...
#
# I provided the function header.  You should fill out the function body,
# including the return statement.
# (define a function)

#def prob_cond_given_pos_bayes(prob_cond, prob_pos_given_cond, prob_neg_given_no_cond):
    # YOUR CODE HERE

#@ 17
# How does the probability of having a condition given you
# tested positive for it change based on how rare the 
# condition is?  
# Produce a histogram showing the probability of having measles
# given you tested positive for measles.  Compute 
# prob_cond_given_pos_bayes(x, 0.98, 0.95) for x ranging
# from 0.001 to 0.10 (x is the probability of having the 
# condition).  Use at least 100 values of x.
# Plot the results as a scatter plot, with x on the x axis
# and probability on the y axis.  Label the x and y axes
# appropriately.  Use matplotlib.
# Assume function prob_cond_given_pos_bayes() is defined.
# (produce a plot)
    
