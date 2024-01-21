# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:36:09 2023

@author: 张浩文
"""

import numpy as np
from MSM import *
from get_data import *


def compute_overlapping(true_res,obtained_res):
    #Computing accuracy
    [n,k] = true_res.shape
    count = 0
    for i in range(n):
        set1 = set(true_res[i,:])
        set2 = set(obtained_res[i,:])
        temp_overrlapping = len(set1 & set2)
        count = count + temp_overrlapping
    return count*1.0 / (n*k)
   