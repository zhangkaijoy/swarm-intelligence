#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:23:06 2018

@author: zhangkai
"""

from numpy import *

dim = 30
maxeval = 300000
lb = -100 * ones((dim,1))
ub = 100 * ones((dim,1))
n = 300; Cr = 0.9; Ca = 1.2 #parameters
A = ub - lb
x = random.rand(dim,1) * (ub - lb) + lb

#sphere function
def f(x): 
    return sum(x*x,0)

fx = f(x)  
eval = 1
while eval < maxeval:
    s = (random.rand(dim,n) * 2 - 1) * tile(A,(1,n)) + tile(x,(1,n))
    
    #boundary handling
    for i in range(dim):
        index = logical_or(s[i,:] > ub[i], s[i,:] < lb[i])
        s[i,index] = random.rand(1,sum(index)) * (ub[i] - lb[i]) + lb[i]; 
        
    fs = f(s)
    eval = eval + n
    if min(fs) < fx:
        x = s[:, argmin(fs)].reshape(dim, 1)
        fx = min(fs)
        A = A * Ca
    else:
        A = A * Cr
        
print(fx)

