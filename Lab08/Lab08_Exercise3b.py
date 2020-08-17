# -*- coding: utf-8 -*-
"""
Chess: exercise 3a
"""
#---------------------- Import modules
from PIL import Image             # Python Imaging Library
import numpy as np                # Numerical Python 
import matplotlib.pyplot as plt   # Python plotting

#---------------------- Create matrix 
n = 500
d = 10

circles = np.zeros((n,n))
cx = n/2
cy = n/2


r1 = d
r2 = 2*d
while r2 < cx: 
    for i in range(n):
        for j in range(n):
            value = (j-cx)**2 + (i-cy)**2 
            if value > r1**2 and value < r2**2:
                circles[i,j] = 1.
    r1 += 2*d  
    r2 += 2*d           
            

#---------------------- Plot image
plt.imshow(circles,cmap= 'gray')      
plt.show()      
#---------------------- Save image
circles2 = circles*255 # to range [0,255]
circles3 = circles2.astype(np.uint8)
Im1 = Image.fromarray(circles3)
Im1.save('circles.jpg')




#%% Partially vectorized code


#---------------------- Import modules
from PIL import Image             # Python Imaging Library
import numpy as np                # Numerical Python 
import matplotlib.pyplot as plt   # Python plotting

#---------------------- Create matrix 
n = 500
d = 10

circles = np.zeros((n,n),dtype=bool)
cx = n/2
cy = n/2

[j, i] = np.meshgrid(np.arange(n),np.arange(n))

r1 = d
r2 = 2*d
while r2 < cx: 
    z1 = (j-cx)**2 + (i-cy)**2 > r1**2
    z2 = (j-cx)**2 + (i-cy)**2 < r2**2
    circles += z1*z2
    r1 += 2*d  
    r2 += 2*d                      

#---------------------- Plot image
plt.imshow(circles,cmap= 'gray')      
plt.show()    
#---------------------- Save image
circles2 = circles*255 # to range [0,255]
circles3 = circles2.astype(np.uint8)
Im1 = Image.fromarray(circles3)
Im1.save('circles.jpg')