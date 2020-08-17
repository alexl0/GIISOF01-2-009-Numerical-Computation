# -*- coding: utf-8 -*-
"""
Lagrange interpolation polynomial
"""
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt
#---------------------- Function 1
def lagrange_fundamental(x,z,k):
    n = len(x)
    l = 1.
    for i in range(n):
        if i != k:
            l *= (z-x[i])/(x[k]-x[i])
    return l
#---------------------- Function 2
def lagrange_polynomial(x,y,z):
    n = len(x)
    p = 0.
    for i in range(n):
        p += y[i]*lagrange_fundamental(x,z,i)
    return p
#----------------------- Data 
x = np.array([2.,3.,4.,5.,6.])
y = np.array([2.,6.,5.,5.,6.])

x1 = np.array([0.,1.,2.,3.,4.,5.,6.])
y1 = np.array([3.,5.,6.,5.,4.,4.,5.])

#------------ Call the function 
xp = np.linspace(min(x),max(x))
yp = lagrange_polynomial(x,y,xp)

xp1 = np.linspace(min(x1),max(x1))
yp1 = lagrange_polynomial(x1,y1,xp1)
#----------------------  Plot 1
plt.plot(x,y,'ro',label = 'nodes')
plt.plot(xp,yp,label = 'polynomial')
plt.legend()
plt.show()    

#----------------------  Plot 2
plt.plot(x1,y1,'ro',label = 'nodes')
plt.plot(xp1,yp1,label = 'polynomial')
plt.legend()
plt.show()
