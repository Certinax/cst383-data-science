# -*- coding: utf-8 -*-
"""
Test code for the random variable code

@author: Glenn
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

from . import PMF, RV

# probability mass functions

pmf = PMF(np.array([1,2,3]), np.array([0.5, 0.1, 0.4]))

pmf.sample(4)

pmf.expect()

pmf.variance()

pmf.prob(np.array([1]))

pmf.prob(np.array([1,3]))

pmf.prob(np.array([4]))

pmf.plot()

# McDonald's example

pmf = PMF(np.array([250, 540, 750]), np.array([0.2, 0.5, 0.3]))

pmf.plot()

pmf.expect()

np.sqrt(pmf.variance())

pmf.prob(np.array([250, 540]))

# Random variables

X = RV(np.array([[0,0], [0,1], [1,0], [1,1]]), np.full(4, 0.25), np.array([0,1,1,2]))
Y = RV(np.array([[0,0], [0,1], [1,0], [1,1]]), np.full(4, 0.25), np.array([0,1,1,0]))

X.sample(5)

X.expect()

X.variance()

X.plot()

X.prob(np.array([2]))

X.prob(np.array([1,2]))

X.event(np.array([1]))

X.event(np.array([2]))

X.cond_prob(np.array([1]), np.array([True, True, True, False]))

X.cond_prob(np.array([2]), np.array([True, True, True, False]))

X.cond_prob(np.array([1]), X.event(np.array([1,2])))

Z = X.add(Y)

Z.variance()

X2 = X.apply_fun(lambda x: x**2)

X2.variance()

# random variable X is the sum of two dice rolls

n = 6
rolls = list(itertools.product(range(6), range(6)))
outcomes = np.array(rolls).reshape((-1,2))+1
probs = np.full(n*n, 1/(n*2))
vals = np.apply_along_axis(np.sum, 1, outcomes)
x = RV(outcomes, probs, vals)

x.plot()

x.expect()

x.variance()

x.prob(np.array([2]))

x.prob(np.array([7]))

x.prob(np.array(list(range(2,11))))

x.cond_prob(np.array([2]), x.event(np.array([1,2,3])))


