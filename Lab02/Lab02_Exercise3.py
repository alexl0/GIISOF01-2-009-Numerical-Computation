# -*- coding: utf-8 -*-
"""
Derivadas sucesivas
"""
#---------------------- Import modules
import numpy as np

#---------------------- Function (1)
def horner(p,x0):
    q = np.zeros_like(p)
    q[0] = p[0]
    for i in range(1,len(p)):
        q[i] = q[i-1]*x0 + p[i]
    return q

#---------------------- Function (2)
def polDer(p,x0):
    der = np.zeros_like(p)    
    factorial = 1.
    q = np.copy(p)
    
    q = horner(q,x0)
    der[0] = q[-1]
    for k in range(1,len(p)):  
        q = horner(q[:-1],x0)
        der[k] = q[-1]*factorial
        factorial *= k+1
    return der    

#----------------------- Data
p = np.array([1, -1, 2, -3,  5, -2])
r = np.array([ 5, -3,  1, -1, -4,  0,  0,  3])
x0 = 1.
x1 = -1.

#------------ Call the function (1)
print('Derivatives of P in x0 = 1')    
print(polDer(p,x0)) 
print('\n')

#------------ Call the function (2)
print('Derivatives of R in x1 = -1')    
print(polDer(r,x1))  
