#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discrete polynomial approximation
"""
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt
#---------------------- Printing options
np.set_printoptions(precision = 2)   # only 2 fractionary digits
np.set_printoptions(suppress = True) # do not use exponential notation

#---------------------- Function 1
def Vandermonde2(x,degree):
    n = len(x)  
    V = np.ones((n,degree+1))
    for j in range(1,degree+1):
        V[:,j] = x**j
    return V  
#---------------------- Function 2
def approximate1(x,y,degree):  
    V = Vandermonde2(x,degree)
    A = np.dot(V.T,V)
    b = np.dot(V.T,y)
    p = np.linalg.solve(A,b)
        
    # check partial results
    if degree == 2:
        print('Coefficient matrix\n')
        print(A)
        print('\nRight hand side matrix\n')
        print(b)
        print('\nSystem solution\n')
        print(p)
   
    return p[::-1] 
#----------------------- Data 
f1 = lambda x: np.cos(x)
x1 = np.linspace(-1,1,5)
y1 = f1(x1)

f2 = lambda x: np.cos(np.arctan(x)) - np.exp(x**2) * np.log(x+2)
x2 = np.linspace(-1,1,10)
y2 = f2(x2)

#------------ Call the function
p1 = approximate1(x1,y1,2)
p2 = approximate1(x2,y2,4)

#----------------------  Plot 1
xp1 = np.linspace(min(x1),max(x1))
yp1 = np.polyval(p1,xp1)

plt.plot(x1,y1,'ro',label = 'nodes')
plt.plot(xp1,yp1,label = 'approximating polynomial')
plt.legend()
plt.show()
    
#----------------------  Plot 2
xp2 = np.linspace(min(x2),max(x2))
yp2 = np.polyval(p2,xp2)

plt.plot(x2,y2,'ro',label = 'nodes')
plt.plot(xp2,yp2,label = 'approximating polynomial')
plt.legend()
plt.show()
