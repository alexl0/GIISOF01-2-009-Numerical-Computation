#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Horner for a vector
"""
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt

#---------------------- Function
def hornerV(p,x):
    y = np.zeros_like(x)
    q = np.zeros_like(p)
    
    for k in range(len(x)):
        x0 = x[k]
        
        q[0] = p[0]
        for i in range(1,len(p)):
            q[i] = q[i-1]*x0 + p[i]
        
        y[k] = q[-1]   
    return y   

#----------------------- Data
p = np.array([1, -1, 2, -3,  5, -2])
r = np.array([ 5, -3,  1, -1, -4,  0,  0,  3])
x = np.linspace(-1,1)

#------------ Call the function (1)
y1 = hornerV(p,x)

#------------ Plot (1)
plt.plot(x,y1)
plt.title('P')
plt.show()

#------------ Call the function (2)
y2 = hornerV(r,x)

#------------ Plot (2)
plt.plot(x,y2)
plt.title('R')
plt.show()
