# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

print("Hello world!")

x = np.array([4,2,1,8])

print(x)

# Print first element
print(x[0])

# Print length of x
print(len(x))

# Print sum of el's in x
print(sum(x))

# Subract 1 from every element
for i in x:
    print(i-1)
    
# Square each element in the array
for i in x:
    print(i**2)