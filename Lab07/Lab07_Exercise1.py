# -*- coding: utf-8 -*-
"""
first derivatives 1
"""
#---------------------- Import modules
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
#---------------------- Function 
def derivatives(f,a,b,h):
    x = np.arange(h,b,h)
    df_f = np.zeros_like(x)
    df_c = np.zeros_like(x)
    df_b = np.zeros_like(x)
    k = 0
    for x0 in x:
        df_f[k] = (f(x0+h) - f(x0))   /h
        df_c[k] = (f(x0+h) - f(x0-h)) /(2*h)
        df_b[k] = (f(x0)   - f(x0-h)) /h
        k += 1
    return x, df_f, df_c, df_b

#----------------------  Plot    
def plotDer(f,df_e,a,b,h):
    # derivatives
    x, df_f, df_c, df_b = derivatives(f,a,b,h) 
    plt.plot(x,df_e(x),label='exact')
    plt.plot(x,df_f,label='forward')
    plt.plot(x,df_b,label='backward')    
    plt.plot(x,df_c,label='centered')
    plt.legend()
    plt.title(r'Derivatives of e$^x$ with h = '+str(h))
    plt.show()
    
    # errors
    plt.plot(x,abs(df_f-df_e(x)),label='forward')
    plt.plot(x,abs(df_b-df_e(x)),label='backward')    
    plt.plot(x,abs(df_c-df_e(x)),label='centered')
    plt.legend()
    plt.title('Errors for h = '+str(h))
    plt.show()
    
    # Print error
    Ef = norm(df_e(x)-df_f)/norm(df_e(x))
    Eb = norm(df_e(x)-df_b)/norm(df_e(x))
    Ec = norm(df_e(x)-df_c)/norm(df_e(x))
    print('\n\nGLOBAL ERRORS\n')
    print('%15s %15s %15s %15s\n' % 
          ('h','E(df_f)','E(df_b)','E(df_C)'))
    print('%15.3f %15.6e %15.6e %15.6e\n' 
          % (h,Ef,Eb,Ec))

#----------------------- Data 
f = lambda x: np.exp(x)
df_e = lambda x: np.exp(x)
a = 0.
b = 1.
h1 = 0.1
h2 = 0.01
#------------ Call the function     
plotDer(f,df_e,a,b,h1)  
plotDer(f,df_e,a,b,h2)

#%%