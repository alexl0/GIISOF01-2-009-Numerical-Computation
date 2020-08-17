# -*- coding: utf-8 -*-
"""
Newton's method
"""
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt

#---------------------- Function 
def newton(f,df,x0,tol=1.0e-12,maxiter=100):
    error = np.inf
    i = 0
    
    while error > tol and i < maxiter:
        x1 = x0 - f(x0)/df(x0)

        i += 1
        error = np.abs(x1-x0)
        x0 = x1
    
    return x1,i 

#----------------------- Data 
f  = lambda x : x**3 - 10*x**2 + 5   
df = lambda x : 3*x**2 - 20*x   
# from incremental search results     
x0 = np.array([-0.7,0.7,9.9])

#------------ Call the function
r = np.zeros(3)

for i in range(3):
    r[i], it = newton(f,df,x0[i])
    print (r[i], it)

#----------------------  Plot
x = np.linspace(min(r)-1,max(r)+1)
plt.plot(x,f(x),x,0*x,'k',r,0*r,'ro')
plt.show()
       
        

