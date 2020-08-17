# -*- coding: utf-8 -*-
"""
Lagrange fundamental polynomials
"""
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt
#---------------------- Function
def lagrange_fundamental(x,z,k):
    n = len(x)
    l = 1.
    for i in range(n):
        if i != k:
            l *= (z-x[i])/(x[k]-x[i])
    return l
#----------------------- Data 
x = np.array([2.,3.,4.,5.,6.])
y = np.array([2.,6.,5.,5.,6.])

#------------ Call the function and Plot  
z = np.linspace(min(x),max(x))
n = len(x) 
ynodes = np.eye(n)
for k in range(n):
    plt.plot(z,lagrange_fundamental(x,z,k))
    plt.plot(x,ynodes[k,:],'o')
    plt.plot(z,0*z,'k')
    plt.title('L'+str(k))
    plt.show()    

