# -*- coding: utf-8 -*-
"""
Incremental Search
"""
#---------------------- Function
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
     
#----------------------- Data 
f = lambda x: x**3 - 10.0*x**2 + 5.0
a = -15.; b = 15.; dx = 0.1

#------------ Call the function
x0, x1 = incrementalSearch(f,a,b,dx)
while x0 != None:
    print("There is a zero in [%.1f, %.1f]" % (x0,x1)) 
    x0, x1 = incrementalSearch(f,x1,b,dx)

