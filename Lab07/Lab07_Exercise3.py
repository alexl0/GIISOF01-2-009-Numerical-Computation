# -*- coding: utf-8 -*-
"""
second derivative
"""
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
#---------------------- Function 
def derivative2(f,a,b,h):
    x = np.arange(h,b,h)
    d2f = np.zeros_like(x)
    
    k = 0
    for x0 in x:
        d2f[k] = (f(x0+h) - 2* f(x0) + f(x0-h)) / h**2
        k += 1
    return x, d2f

#----------------------  Plot    
def plotDer2(f,d2f_e,a,b,h):
    # derivatives
    x, d2f_a = derivative2(f,a,b,h) 
    
    #plot
    plt.plot(x,d2f_e(x),'y',linewidth=6,label='exact')
    plt.plot(x,d2f_a,'b--',label='approximate')
    plt.legend()
    plt.title(r'Second derivative of $\sin(2 \pi x)$')
    plt.show()
    
    
    # Print error
    E = norm(d2f_e(x)-d2f_a)/norm(d2f_e(x))

    print('%15s %15s\n' % ('h','E(df_f)'))
    print('%15.3f %15.6e\n' % (h,E))

#----------------------- Data 
f = lambda x: np.sin(2*np.pi*x)
d2f_e = lambda x: -(2*np.pi)**2 * np.sin(2*np.pi*x)
a = 0.
b = 1.
h = 0.01
#------------ Call the function     
plotDer2(f,d2f_e,a,b,h)  

#%%