# -*- coding: utf-8 -*-
"""
Horner for a point
"""
#---------------------- Import modules
import numpy as np

#---------------------- Function 
def horner(p,x0):
    q = np.zeros_like(p)
    q[0] = p[0]
    for i in range(1,len(p)):
        q[i] = q[i-1]*x0 + p[i]
    return q

#----------------------- Data
p = np.array([1, -1, 2, -3,  5, -2])
r = np.array([ 5, -3,  1, -1, -4,  0,  0,  3])
x0 = 1.
x1 = -1.

#------------ Call the function (1)
q = horner(p,x0)
print('Q coefficients = ', q[:-1])
print('P(1) = ', q[-1])
print('\n')

#------------ Call the function (2)
q1 = horner(r,x1)
print('Q1 coefficients = ', q1[:-1])
print('R(-1) = ', q1[-1])
