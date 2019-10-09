#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:40:04 2019

@author: certinax
"""

import numpy as np
import pandas as pd


# Problem 1
infile = "https://raw.githubusercontent.com/grbruns/cst383/master/campaign-ca-2016-sample.csv"

df = pd.read_csv(infile)

df.info()

# Problem 2

# Problem 3
# 3 columns are markes as numeric, file_num, contb_receipt_amt, contbr_zip
# cmte_id and cand_id could easily be numeric when removing leading char (C/P)

# Problem 4

