# -*- coding: utf-8 -*-
"""

Python classes to implement probability mass functions and 
discrete random variables.

Please supply code wherever you see "YOUR CODE HERE".
Do not modify the file in any other way!

In my code I used no loops, and the bodies of most methods
were just 1 or 2 lines.

"""

import numpy as np
import matplotlib.pyplot as plt

###################################################################
#
# PMF
#
###################################################################

# A PMF object models a probability mass function with finite support.
# Such an object has two attributes: "support", which is a numeric
# array, and "probs", another numeric array of the same length as
# "support".  The idea is that the probability of value support[0]
# is probs[0], the probability of support[1] is probs[1], etc.
# 
# So if we have a probability mass function with support 
# np.array([1,2,3]), we might have probs c(0.5, 0.1, 0.4), meaning 
# that the probability of 1 is 0.5, the probability of 2 is 0.1, 
# and the probability of 3 is 0.4.
#
# In the comments below, I will use object 'pmf' for this example.

class PMF:
    # Return a new PMF object.
    # support - a non-empty numberic NumPy array
    # probs - a NumPy array of non-negative numbers, with
    #  the same length as support.  probs is normalized to 
    #  ensure it sums to 1.
    # example: pmf = PMF(np.array([1,2,3], np.array([0.5, 0.1, 0.4])))
    def __init__(self, support, probs):
        if support.size == 0 or support.size != probs.size:
            print('error: support must be non-empty and equal in length to probs')
            exit(1)
        
        # normalize probs so they sum to 1
        probs = probs/probs.sum()
        
        # put support and probs in increasing order of support
        idx = support.argsort()
        self.support = support[idx]
        self.probs = probs[idx]
    
#@ 1
    # return an array of n samples from this pmf (with replacement)
    # example: pmf.sample(4) might give 1, 2, 1, 3
    def sample(self, n=1):
        return np.random.choice(self.support, n, p=self.probs)
    
#@ 2
    # return the expectation of this pmf
    # example: pmf.expect() gives 1.9
    def expect(self):
        # YOUR CODE HERE
        # hint: do not use sampling; return an exact answer
        return sum(self.support * self.probs)
#@ 3
    # return the variance of this pmf
    # example: pmf.variance() gives 0.89
    def variance(self):
        # YOUR CODE HERE
        # hint: do not use sampling; return an exact answer
        return sum(((self.support - self.expect())**2)*self.probs)
#@ 4
    # return the probability associated with the subset
    # x of the support for this PMF
    # the subset x of the support for this PMF
    # example: pmf.prob(np.array([1])) gives 0.5
    #   (in other words, the probability of 1 is 0.5)
    # example: pmf.prob(np.array([1,3])) gives 0.9
    #   (in other words, the probability of 1 or 3 is 0.9)
    # example: pmf.prob(np.array([4])) gives 0
    #   (code should handle x that is not a subset of the support)
    def prob(self, x):
        # YOUR CODE HERE
        # hint: consider the use of function numpy.in1d()
        return sum(self.probs[np.isin(self.support, x)])
#@ 5
    # plot the PMF as a bar plot
    # example: pmf.plot() gives three bars, with labels 1,2,3
    # and heights 0.5, 0.1, 0.4.
    def plot(self):
        # YOUR CODE HERE
        # hint: I recommend using plt.bar()
        bar_pos = np.arange(len(self.support))
        plt.bar(bar_pos, self.probs, color='darkgreen')
        plt.xticks(bar_pos, self.support)
        plt.xlabel("Support values")
        plt.ylabel("Corresponding probabilities")
        plt.title("PMF")
#@      
        
        
        
        
pmf = PMF(np.array([1,2,3]), np.array([0.5, 0.1, 0.4]))
#print(pmf.expect())
#print(pmf.variance())
#print(pmf.prob(np.array([2])))
#pmf.plot()

###################################################################
#
# RV
#
###################################################################

# An RV object models a discrete random variable with finite support.
# This means there are only finitely-many values the random variable
# can take.  In an RV object we capture a sample space, the values
# that the random variable gives for each outcome in the sample
# space, and the probability of each outcome.
#
# In this code, an RV object has three attributes: "outcomes", which 
# is a numpy matrix with as many rows as there are outcomes in the sample
# space, "probs", a numeric array that gives a probability for each row
# of the outcomes data frame, and "vals", which is a numeric array which
# gives a number for each outcome.  Remember that a random variable is
# defined as something that labels each outcome of a sample space with a
# number.  That is what "vals" does.
# 
# So if we have a random variable X that gives the number of heads in
# two coin flips, 
# outcomes could be the matrix np.array([[0,0],[0,1],[1,0],[1,1]]), 
# probs could be np.array([0.25, 0.25, 0.25, 0.25]), and
# vals could be np.array([0, 1, 1, 2]).
#
# In the comments below, I use 'X' for this example.  I use 'X' for the
# random variable that is 1 when exactly one coin is head, and 0 otherwise.
        
class RV:

    # return a discrete random variable with finite support
    # outcomes - a NumPy matrix of n > 0 rows; each row represents one outcome
    # probs    - probability of each outcome; a numeric NumPy array of length n, 
    #            all values >= 0
    # vals     - numeric value of each outcome; a numeric NumPy array of length n
    # example: X = rv_create(np.array([[0,0],[0,1],[1,0],[1,1]]),
    #                        probs=np.array([0.25, 0.25, 0.25, 0.25]),
    #                        vals=np.array([0,1,1,2]))
    # example: Y = rv_create(np.array([[0,0],[0,1],[1,0],[1,1]]),
    #                        probs=np.array([0.25, 0.25, 0.25, 0.25]),
    #                        vals=np.array([0,1,1,0]))
    def __init__(self, outcomes, probs, vals):
        n = outcomes.shape[0]
        if n == 0 or probs.size != n or not np.all(probs >= 0):
            print('error: sample space must be non-empty and each outcome must have a non-negative prob')
            exit(1)
        
        self.outcomes = outcomes
        self.vals = vals
        self.probs = probs/probs.sum()    # the probs must sum to 1
        
        # compute the PMF
        support = np.unique(vals)
        val_probs = np.array([self.probs[self.vals == x].sum() for x in support])
        self.pmf = PMF(support, val_probs)
    
#@ 6
    # return n samples from this random variable
    # example: X.sample(5) might give 1, 2, 1, 0, 1
    def sample(self, n=1):
        # YOUR CODE HERE
        # hint: would the PMF sample() method be handy here?
        return np.random.choice(self.vals, n, p=self.probs)
    
#@ 7
    # return the expectation of this random variable
    # example: X.expect() is 1
    def expect(self):
        # YOUR CODE HERE
        # hint: my code for this is just one short line
        return sum(self.vals * self.probs)

#@ 8
    # return the variance of this random variable
    # example: X.variance() is 0.5
    def variance(self):
        # YOUR CODE HERE
        return sum(((self.vals - self.expect())**2)*self.probs)

#@ 9
    # return the probability that the value of this random variable
    # is in the set of values defined by NumPy array a
    # example: X.prob(np.array([2])) is 0.25
    # example: X.prob(np.array([1,2])) is 0.75
    def prob(self, a):
        # YOUR CODE HERE
        # hint: remember that RV objects have a PMF component
        return sum(self.probs[np.isin(self.vals,a)])
    
#@ 10
    # return an NumPy boolean mask giving True for the 
    # outcomes associated with the subset a of the support of 
    # the random variable (and False for other outcomes)
    # example: X.event(np.array([1])) is False, True, True, False  
    #   (because in X, the second and third row of the outcomes 
    #    matrix are associated with value 1)
    # example: X.event(1) is False, True, True, False
    # example: X.event(np.array([2])) is False, False, False, True
    # example: X.event(np.array([0,1,2])) is True, True, True, True
    # read about numpy.in1d()
    def event(self, a):
        # YOUR CODE HERE
        return np.isin(self.vals, a)

#@ 11
    # return the conditional probability that the value of this
    # random variable belongs to a set of values, given some
    # condition.  The set of values is given as a NumPy array
    # of values that are contained in the support of this random
    # variable.  The condition is given as a boolean array
    # indicating the elements of the sample space that define
    # the condition.
    # example: X.cond_prob(np.array([1]), np.array([True, True, True, False])) is 0.6667
    # example: X.cond_prob(np.array([2]), np.array([True, True, True, False])) is 0
    # example: X.cond_prob(np.array([1]), X.event(np.array([1,2])) is 0.6667
    def cond_prob(self, a, event):
        # YOUR CODE HERE
        # hint: my code contains a call to the constructor for RV
        return self.vals[event][self.vals[event] == a].size/self.vals[event].size

#@ 12
    # return a new random variable that is the sum of this
    # and another random variable Y
    # Y must have the same sample space as this variable.
    # example: Z = X.add(Y)
    # example: Z.variance() is 0.75   
    def add(self, Y):
        if not np.array_equal(self.outcomes, Y.outcomes)  \
            or not np.array_equal(self.probs, Y.probs):
            print('error: this and Y must share the same sample space and probabilities')
            exit(1)
        # YOUR CODE HERE
        return RV(self.outcomes, self.probs, self.vals + Y.vals)
    
#@ 13
    # return a new random variable by applying a function f to this random variable
    # example: X2 = X.apply(lambda x: x**2)
    # example: X2.variance() is 2.25
    def apply_fun(self, f):
        # YOUR CODE HERE
        # hint: my code contains a call to the constructor for RV
        return RV(self.outcomes, self.probs, f(self.vals))
    
#@ 14
    # plot the PMF associated with this random variable
    # example: X.plot()
    def plot(self):
        # YOUR CODE HERE
        bar_pos = np.arange(len(self.vals))
        plt.bar(bar_pos, self.probs, color='darkgreen')
        plt.xticks(bar_pos, self.vals)
        plt.xlabel("Values")
        plt.ylabel("Corresponding probabilities")
        plt.title("PMF")