#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:12:23 2019

@author: certinax
"""

import numpy as np

x = np.array([4, 6, 2, 4, 9, 5])
splits = [1,4]
print(np.split(x, splits))
print(x)

y = np.array([20, 10, 30, 40])
print(y > 20)
mask = y > 20
print(y[mask])

x = np.array([4, 6, 2, 4, 9, 5])
print(x[[0,3,4,5]])
print(x[x[2:4]])
print(x[2:4])

x = np.array([range(i, i+3) for i in [2,4,6]])
print(x)
print(x[1,2])
print(x[1,])



def one_thousand_weighted_coin_tosses():
    return np.random.choice(2, 1000, p=[0.1, 0.9])

def generate_fraction_array_1000(y2):
    for i, val in enumerate(y2):
        coin = one_thousand_weighted_coin_tosses()
        frac = len(coin[coin == 1]) / len(coin)
        y2[i] = frac
    return y2

y2 = np.empty(100)
y2 = generate_fraction_array_1000(y2)


def two_hundred_weighted_coin_tosses():
    return np.random.choice(2, 200, p=[0.1, 0.9])


def generate_fraction_array(y1):
    for i, val in enumerate(y1):
        coin = two_hundred_weighted_coin_tosses()
        frac = len(coin[coin == 1]) / len(coin)
        y1[i] = frac
    return y1

y1 = np.empty(100) # Uninitialized array with size = 100

y1 = generate_fraction_array(y1) # Generating the fractions of 1's of 200 coin tosses
                                 # for each element in array y1


ax1 = plt.subplot(3,1,1)
ax1.hist(y1)
ax1.set_title("Plot for Y1")
ax1.set_xlim([0.85, 0.95])

ax2 = plt.subplot(3,1,3)
ax2.hist(y2)
ax2.set_title("Plot for Y2")
ax2.set_xlim([0.85, 0.95])



prob_func = np.array([prob_cond_given_pos_bayes(x[i], 0.98, 0.95) for i in range(100)])

#fig, ax = plt.subplots(1)
## Option 1
#ax[0].hist(prob_func)
#ax[0].scatter(x, prob_cond_given_pos_bayes(x, 0.98, 0.95))

## Option 2
#ax[0].hist([prob_cond_given_pos_bayes(x[i], 0.98, 0.95) for i in range(100)])
#ax[1].scatter(x, [prob_cond_given_pos_bayes(x[i], 0.98, 0.95) for i in range(100)])

#ax[0].set_xlabel("Probability of having condition")
#ax[0].set_ylabel("Probability of condition if tested positive")