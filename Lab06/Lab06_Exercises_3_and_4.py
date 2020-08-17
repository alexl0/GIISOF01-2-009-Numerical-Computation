# -*- coding: utf-8 -*-
"""
Continuous polynomial approximation
"""
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

#---------------------- Printing options
np.set_printoptions(precision = 2)   # only 2 fractionary digits
np.set_printoptions(suppress = True) # do not use exponential notation
 
#---------------------- Function
def approximate2(f,a,b,degree):
    
    A = np.zeros((degree+1,degree+1))
    c = np.zeros(degree+1)
    
    for i in range(degree+1):
        
        g = lambda x: x**i*f(x)
        c[i] = quad(g,a,b)[0]
        
        for j in range(i,degree+1):
            g = lambda x: x**(i+j)
            A[i,j] = quad(g,a,b)[0]
           
            # matrix A is symmetric
            if i != j:
                A[j,i] = A[i,j]
    
    p = np.linalg.solve(A,c)
        
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
a1 = -1.
b1 = 1.
degree1 = 2

f2 = lambda x: np.cos(np.arctan(x)) - np.exp(x**2) * np.log(x+2)
a2 = -1.
b2 = 1.
degree2 = 4

#------------ Call the function
p1 = approximate2(f1,a1,b1,degree1)
p2 = approximate2(f2,a2,b2,degree2)

#----------------------  Plot 1
xp1 = np.linspace(a1,b1)
yp1 = np.polyval(p1,xp1)

plt.plot(xp1,f1(xp1),label = 'function')
plt.plot(xp1,yp1,label = 'approximating polynomial')
plt.legend()
plt.show()

#----------------------  Error 1
Er1 = np.linalg.norm(f1(xp1)-yp1)/np.linalg.norm(f1(xp1))
print('Er1 = ', Er1)

#----------------------  Plot 2
xp2 = np.linspace(a2,b2)
yp2 = np.polyval(p2,xp2)

plt.plot(xp2,f2(xp2),label = 'function')
plt.plot(xp2,yp2,label = 'approximating polynomial')
plt.legend()
plt.show()

#----------------------  Error 1
Er2 = np.linalg.norm(f2(xp2)-yp2)/np.linalg.norm(f2(xp2))
print('Er2 = ', Er2)
