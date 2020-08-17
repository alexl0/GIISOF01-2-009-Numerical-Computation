# -*- coding: utf-8 -*-
"""
Vandermonde's matrix and interpolation
"""
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt
#---------------------- Function 1
def Vandermonde(x):
    n = len(x)  
    V = np.ones((n,n))
    for j in range(1,n):
        V[:,j] = x**j
    return V     
#---------------------- Function 2
def polVandermode(x,y):
    V = Vandermonde(x)
    p = np.linalg.solve(V,y)
    p = p[::-1]
    return p
#----------------------- Data 
x = np.array([2.,3.,4.,5.,6.])
y = np.array([2.,6.,5.,5.,6.])

x1 = np.array([0.,1.,2.,3.,4.,5.,6.])
y1 = np.array([3.,5.,6.,5.,4.,4.,5.])

#------------ Call the function
p = polVandermode(x,y)
p1 = polVandermode(x1,y1)

#----------------------  Plot 1
xp = np.linspace(min(x),max(x))
yp = np.polyval(p,xp)

plt.plot(x,y,'ro',label = 'nodes')
plt.plot(xp,yp,label = 'polynomial')
plt.legend()
plt.show()
    
#----------------------  Plot 2
xp1 = np.linspace(min(x1),max(x1))
yp1 = np.polyval(p1,xp1)

plt.plot(x1,y1,'ro',label = 'nodes')
plt.plot(xp1,yp1,label = 'polynomial')
plt.legend()
plt.show()
