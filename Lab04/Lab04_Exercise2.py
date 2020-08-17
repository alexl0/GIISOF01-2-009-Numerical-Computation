# -*- coding: utf-8 -*-
"""
Fixed point method
"""
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt
#---------------------- Function 1
def incrementalSearch(f,a,b,dx):
    x0 = a
    x1 = a + dx
    while (x1 < b):
        if f(x0)*f(x1) < 0:
            return x0, x1
        else:
            x0 += dx
            x1 += dx
    return None, None

#---------------------- Function 2
def fixedPoint(g,x0,tol=1.0e-12,maxiter=200):
    error = np.inf
    i = 0
    
    while error > tol and i < maxiter:
        x1 = g(x0)

        i += 1
        error = np.abs(x1-x0)
        x0 = x1
    
    return x1,i 
     
#----------------------- Data 1
f = lambda x: np.exp(-x) - x
g = lambda x: np.exp(-x)
a = -3.; b = 3.; dx = 0.1

#------------ Call the functions 1
x0, x1 = incrementalSearch(f,a,b,dx)
r, i = fixedPoint(g,x0)

#------------ Print the solutions 1
print("There is a zero in [%.1f, %.1f]" % (x0,x1))
print(r,i)

#----------------------  Plot 1
x = np.linspace(0,1)
plt.plot(x,g(x),'r',label = 'g')
plt.plot(x,x,'b',label = 'y = x')
plt.plot(r,r,'bo')
plt.legend()
plt.show()

#----------------------- Data 2
f = lambda x: x - np.cos(x)
g1 = lambda x: np.cos(x)
g2 = lambda x: 2*x - np.cos(x)
g3 = lambda x: x - (x - np.cos(x))/(1+np.sin(x))
g4 = lambda x: (9*x + np.cos(x))/10
a = -3.; b = 3.; dx = 0.1

#------------ Call the functions 2
x0, x1 = incrementalSearch(f,a,b,dx)

r = np.zeros(4)
it = np.zeros(4,dtype=np.uint8)
i = 0
for g in [g1,g2,g3,g4]:
    r[i], it[i] = fixedPoint(g,x0)
    i += 1

#------------ Print the solutions 2
print("There is a zero in [%.1f, %.1f]" % (x0,x1))
for i in range(len(r)):
    print('g'+str(i+1),r[i],it[i])

#----------------------  Plot 2
x = np.linspace(0,1)
plt.plot(x,g1(x),'r',label = 'g1')
plt.plot(x,g2(x),'m',label = 'g2')
plt.plot(x,g3(x),'g',label = 'g3')
plt.plot(x,g4(x),'y',label = 'g4')
plt.plot(x,x,'b',label = 'y = x')
plt.plot(r[0],r[0],'bo')
plt.legend()
plt.show()
