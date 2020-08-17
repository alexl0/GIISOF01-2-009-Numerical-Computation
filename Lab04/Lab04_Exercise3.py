# -*- coding: utf-8 -*-
"""
Function optimization
"""
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import scipy.optimize as op

#---------------------- Lambda functions
x = sym.Symbol('x', real=True)

f_sim = x**3 + sym.log(x+7) * sym.cos(4*x) - 1
df_sim  = sym.diff(f_sim,x)
d2f_sim = sym.diff(df_sim,x)

f   = sym.lambdify([x], f_sim,'numpy') 
df  = sym.lambdify([x], df_sim,'numpy') 
d2f = sym.lambdify([x], d2f_sim,'numpy') 

#---------------------- Plot f'
x = np.linspace(-2,2)
plt.plot(x,df(x))
plt.plot(x,0*x,'k')
plt.title('f derivative')
plt.show()

#---------------------- Find zeros f'
x0 = np.array([-1.5, -1., 0., 1.])
r = np.zeros_like(x0)
for i in range(len(x0)):
    r[i]= op.newton(df,x0[i],fprime=d2f,tol=1.e-12,maxiter=100)       
    print('x'+str(i+1)+'=',r[i])
    
#---------------------- Plot f with the extrema
maxi = r[[0,2]]
mini = r[[1,3]]
    
x = np.linspace(-2,2)
plt.plot(x,f(x), label = 'function')
plt.plot(maxi,f(maxi),'ro',label = 'Max')
plt.plot(mini,f(mini),'go',label = 'min')
plt.plot(x,0*x,'k')
plt.legend()
plt.show()    
    
#---------------------- Check second derivatives
for i in range(len(r)):
    print('d2f(x'+str(i+1)+') = ',d2f(r[i]))