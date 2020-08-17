# -*- coding: utf-8 -*-
"""
e^x McLaurin series
"""
#%% (1) Approximate e^x in x0
#---------------------- Import modules
import numpy as np

#----------------------- Data 
f = lambda x: np.exp(x)
x0 = -0.5
#----------------------- Script
pol = 0.
error = np.inf
i = 0
factorial = 1.
tol = 1.e-6
maxNumSum = 100

while (error > tol and i < maxNumSum):
    term = x0**i/factorial
    pol += term
    error = abs(term)

    i += 1
    factorial *= i
#----------------------- Print solution    
print('Function value in -0.5      =', f(x0))
print('Approximation value in -0.5 =', pol)
print('Number of iterations        =', i) 

#%% (2) Plot e^x in [-1,1] using a script
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt

#----------------------- Data 
f = lambda x: np.exp(x)
x = np.linspace(-1,1)

#----------------------- Script
pol = 0.
error = np.inf
i = 0
factorial = 1.
tol = 1.e-6
maxNumSum = 100

while (error > tol and i < maxNumSum):
    term = x**i/factorial
    pol += term
    error = abs(np.max(term))

    i += 1
    factorial *= i
    
#----------------------- Plot  
plt.plot(x,f(x),'y',linewidth=4,label='f')
plt.plot(x,pol,'k--',label='f approximation')
plt.plot(x,0*x,'k')
plt.legend()
plt.title('f approximation with McLaurin series')
plt.show()

#%% (3) Plot e^x in [-1,1] using a function
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt

#---------------------- Function 
def funExp(x,tol,maxNumSum):  
    pol = 0.
    error = np.inf
    i = 0
    factorial = 1.
    tol = 1.e-6
    maxNumSum = 100
    
    while (error > tol and i < maxNumSum):
        term = x**i/factorial
        pol += term
        error = abs(np.max(term))
    
        i += 1
        factorial *= i
        
    return pol

#----------------------- Data 
f = lambda x: np.exp(x)
x = np.linspace(-1,1)

#------------ Call the function
y = funExp(x, 1.e-6, 100)

#----------------------  Plot
plt.plot(x,f(x),'y',linewidth=4,label='f')
plt.plot(x,y,'k--',label='f approximation')
plt.plot(x,0*x,'k')
plt.legend()
plt.title('f approximation with McLaurin series')
plt.show() 

