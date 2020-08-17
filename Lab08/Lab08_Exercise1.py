#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lena: exercise 1
"""
#---------------------- Import modules
from PIL import Image             # Python Imaging Library
import numpy as np                # Numerical Python 
import matplotlib.pyplot as plt   # Python plotting

#---------------------- Import image 
I = Image.open('lena_gray_512.tif')
#---------------------- Transform to numpy
a = np.asarray(I,dtype=np.float64)
#---------------------- Create masks
m = a.shape[0]
n = a.shape[1]
mask1 = np.zeros((m,n))
mask2 = np.ones((m,n))*0.5

cx = int(n/2)
cy = int(m/2)

for i in range(m):
    for j in range(n):
        if (j-cx)**2 + (i-cy)**2 < 150**2: 
            mask1[i,j] = 1.
            mask2[i,j] = 1. 
#---------------------- Apply masks
a1 = a*mask1
a2 = a*mask2
#---------------------- Plot images
plt.imshow(a1,cmap= 'gray')      
plt.show()      

plt.imshow(a2,cmap= 'gray')      
plt.show()



#%% Vectorized code



#---------------------- Import modules
from PIL import Image             # Python Imaging Library
import numpy as np                # Numerical Python 
import matplotlib.pyplot as plt   # Python plotting

#---------------------- Import image 
I = Image.open('lena_gray_512.tif')
#---------------------- Transform to numpy
a = np.asarray(I,dtype=np.float64)
#---------------------- Create masks
m = a.shape[0]
n = a.shape[1]

cx = int(n/2)
cy = int(m/2)

[j, i] = np.meshgrid(np.arange(n),np.arange(m))
mask1 = (j-cx)**2 + (i-cy)**2 < 150**2
mask2 = (j-cx)**2 + (i-cy)**2 > 150**2
mask2 = mask2.astype(float)* 0.5 + mask1.astype(float)

#---------------------- Apply masks
a1 = a*mask1
a2 = a*mask2
#---------------------- Plot images
plt.imshow(a1,cmap= 'gray')      
plt.show()      

plt.imshow(a2,cmap= 'gray')      
plt.show()