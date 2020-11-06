#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:12:20 2019

@author: certinax
"""

import numpy as np
import pandas as pd

#df = pd.read_csv("default of credit card clients.csv", delimiter=";", header=1)

#df.info()

#print(df["AGE"].value_counts())

#print(df.iloc[1,:])

df = pd.read_csv("Fire_Incidents.csv", nrows=100000)

df.info()

print(df["Civilian Injuries"].value_counts())