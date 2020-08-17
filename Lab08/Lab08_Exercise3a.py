#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chess: exercise 3a
"""
#%% with concatenate and tile


#---------------------- Import modules
from PIL import Image             # Python Imaging Library
import numpy as np                # Numerical Python 
import matplotlib.pyplot as plt   # Python plotting

#---------------------- Create matrix 
black = np.zeros((250,250))
white = np.ones((250,250))

a1 = np.concatenate((white,black),axis=1)
a2 = np.concatenate((black,white),axis=1)
a  = np.concatenate((a1,a2))

chess = np.tile(a,(4,4))
            
#---------------------- Plot image
plt.imshow(chess,cmap= 'gray')      
plt.show()      

#---------------------- Save image
chess2 = chess*255 # to range [0,255]
chess3 = chess2.astype(np.uint8)
Im1 = Image.fromarray(chess3)
Im1.save('chess.png')


#%% with slicing


#---------------------- Import modules
from PIL import Image             # Python Imaging Library
import numpy as np                # Numerical Python 
import matplotlib.pyplot as plt   # Python plotting

#---------------------- Create matrix 
black = np.zeros((250,250))
chess = np.ones((250*8,250*8))

k = 1
for i in np.arange(8)*250:
    
    if k == 1:
        k = -1
        j = 250
        while j < 250*8:
            chess[i:i+250,j:j+250] = black
            j += 500
    else:
        k = 1
        j = 0
        while j < 250*8:
            chess[i:i+250,j:j+250] = black
            j += 500
            

#---------------------- Plot image
plt.imshow(chess,cmap= 'gray')      
plt.show()      
#---------------------- Save image
chess2 = chess*255 # to range [0,255]
chess3 = chess2.astype(np.uint8)
Im1 = Image.fromarray(chess3)
Im1.save('chess.png')



#%% vectorized code



#---------------------- Import modules
from PIL import Image             # Python Imaging Library
import numpy as np                # Numerical Python 
import matplotlib.pyplot as plt   # Python plotting

#---------------------- Create matrix 
n = 250
chess = np.zeros((8*n,8*n))

[j,i] = np.meshgrid(np.arange(8*n),np.arange(8*n))

I = np.floor(i/n)
J = np.floor(j/n)
mask = np.remainder(I+J,2) == 0 

#---------------------- Plot image
plt.imshow(mask,cmap='gray')
plt.show()

#---------------------- Save image
chess2 = mask*255 # to range [0,255]
chess3 = chess2.astype(np.uint8)
Im1 = Image.fromarray(chess3)
Im1.save('chess.png')