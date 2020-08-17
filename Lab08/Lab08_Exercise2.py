#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lena: exercise 2
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
line = np.linspace(0,1,m)
mask = np.tile(line,(n,1))
mask = mask.T 
#---------------------- Apply masks
a1 = a*mask
#---------------------- Plot images
plt.imshow(a,cmap= 'gray')      
plt.show()      

plt.imshow(a1,cmap= 'gray')      
plt.show()

#%% A different code
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

[j, i] = np.meshgrid(np.arange(n),np.arange(m))
mask = i

#---------------------- Apply masks
a1 = a*mask
#---------------------- Plot images
plt.imshow(a,cmap= 'gray')      
plt.show()      

plt.imshow(a1,cmap= 'gray')      
plt.show()


