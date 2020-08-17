# -*- coding: utf-8 -*-
"""
secant
"""
import numpy as np
import matplotlib.pyplot as plt
#---------------------- Function 
def secant(f,x0,x1,tol=1.0e-12,maxiter=100):
    error = np.inf
    i = 0
    
    while error > tol and i < maxiter:
        x2 = x1 - f(x1) * (x1-x0) / (f(x1)-f(x0))

        i += 1
        error = np.abs(x2-x1)
        x0 = x1
        x1 = x2
    
    return x2,i 

#----------------------- Data 
f  = lambda x : x**3 - 10*x**2 + 5     
# from incremental search results
x0 = np.array([-0.7,0.7,9.9])
x1 = x0 + 0.1

#------------ Call the function
r = np.zeros(3)
for i in range(3):
    r[i], it = secant(f,x0[i], x1[i])
    print (r[i], it)

#----------------------  Plot
x = np.linspace(min(r)-1,max(r)+1)
plt.plot(x,f(x),x,0*x,'k',r,0*r,'ro')
plt.show()
