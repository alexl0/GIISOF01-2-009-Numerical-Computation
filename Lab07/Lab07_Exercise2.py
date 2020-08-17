# -*- coding: utf-8 -*-
"""
first derivatives
"""
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
#---------------------- Function 1
def derivatives_a(f,a,b,h):

    x = np.arange(a,b+h,h)
    df = np.zeros_like(x)
    
    # derivatives at the borders
    df[0]  = (f(a+h) - f(a)) / h
    df[-1] = (f(b) - f(b-h)) / h 
    
    # derivatives in the middle
    k = 1
    for x0 in x[1:-1]:
        df[k] = (f(x0+h) - f(x0-h)) /(2*h)
        k += 1
    return x, df
#---------------------- Function 2
def derivatives_b(f,a,b,h):

    x = np.arange(a,b+h,h)
    df = np.zeros_like(x)
    
    # derivatives at the borders
    df[0]  = (- 3*f(a) + 4*f(a+h) - f(a+2*h)) / (2*h)
    df[-1] = (f(b-2*h) - 4*f(b-h) + 3*f(b)) / (2*h) 
    
    # derivatives in the middle
    k = 1
    for x0 in x[1:-1]:
        df[k] = (f(x0+h) - f(x0-h)) /(2*h)
        k += 1
    return x, df
#----------------------  Plot    
def plotDer_ab(f,df_e,derivatives,a,b,h):
    
    # derivatives
    x, df = derivatives(f,a,b,h) 
    
    # plot
    plt.plot(x,df_e(x),'y',linewidth=6,label='exact')
    plt.plot(x,df,'b--',label='approximate')
    plt.legend()
    plt.title('Derivative of 1/x')
    plt.show()
    
    # Print error
    E = norm(df_e(x)-df)/norm(df_e(x))

    print('%15s %15s\n' % ('h','E(df_f)'))
    print('%15.3f %15.6e\n' % (h,E))

#----------------------- Data 
f = lambda x: 1./x
df_e = lambda x: -1./x**2
a = 0.2
b = 1.2
h = 0.01

#------------ Call the function   
print('(a)')  
plotDer_ab(f, df_e, derivatives_a, a, b, h)  
print('(b)')  
plotDer_ab(f, df_e, derivatives_b, a, b, h)
#%%