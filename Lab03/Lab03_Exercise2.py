# -*- coding: utf-8 -*-
"""
Bisection method
"""
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt

#---------------------- Function 
def bisection(f,a,b,tol=1.0e-12,maxiter=100):
    error = np.inf
    xant = a
    i = 0
    
    while error > tol and i < maxiter:
        x  = (a+b)/2
        
        i += 1
        error = np.abs(x-xant)
        xant = x
        
        if f(a)*f(x) < 0:
            b = x
        elif f(x)*f(b) < 0:
            a = x
        else:
            return x, i
    
    return x,i
#----------------------- Data 
f = lambda x : x**3 - 10*x**2 + 5 
# from incremental search results
# because they meet Bolzano's conditions
a = np.array([-0.7,0.7,9.9])
b = a + 0.1

#------------ Call the function
r = np.zeros(3)
for i in range(3):
    r[i], it = bisection(f,a[i],b[i])
    print (r[i], it)

#----------------------  Plot
x = np.linspace(min(r)-1,max(r)+1)
plt.plot(x,f(x))
plt.plot(x,0*x,'k')
plt.plot(r,0*r,'ro')
plt.show()

       
        

