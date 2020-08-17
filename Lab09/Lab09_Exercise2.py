# -*- coding: utf-8 -*-
"""
holi
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
#%%
np.random.seed(1)
I = Image.open("holi.jpg")
# number of pixels and colors
w, h = I.size
colors = I.getcolors(w * h)
num_colores = len(colors) 
num_pixels = w*h  
 
# convert to numpy matrix
a = np.asarray(I,dtype=np.float32)/255

#%
plt.figure(figsize=(12,12))
plt.imshow(a)
plt.axis('off')
plt.show()

print ('Number of pixels = ', num_pixels)
print ('Number of colors = ', num_colores)

# rearrange and cluster
x, y, z = a.shape
a1 = a.reshape(x*y, z)

n = 10
k_means = KMeans(n_clusters=n)
k_means.fit(a1)

centroids = k_means.cluster_centers_
labels = k_means.labels_

# reshape and show result
a2 = centroids[labels]
a3 = a2.reshape(x,y,z)

plt.figure(figsize=(12,12))
plt.imshow(a3)
plt.axis('off')
plt.show()

print ('Number of pixels = ', num_pixels)
print ('Number of colors = ', n)

#%% create color matrix without sorting

# color matrix without sorting        
block0 = np.ones((100,1000,3))
block = np.zeros((1000,1000,3))
j = 0
for f in np.arange(n)*100:
    for k in range(3):
        block[f:f+100,:,k] = block0[:,:,k]*centroids[j,k]
    j += 1
block2 = block*255
plt.imshow(block)
plt.show()

# save image file
a3 = block2.astype(np.uint8)
Im = Image.fromarray(a3)
Im.save("colors1.jpg")

#%% create color matrix sorting colors by euclidean distance

# Sort colors
colors = np.copy(centroids)
for i in range(n-1):

    a0 = colors[i,:]
    dopt = np.inf
    fila = i+1
    
    for j in range(i+1,n):

        a1 = colors[j,:]
        d = np.linalg.norm(a0-a1)
        if d < dopt:
            dopt = d
            fila = j
        
    temp = np.copy(colors[i+1,:])         
    colors[i+1,:] = (colors[fila,:])
    colors[fila,:] = (temp)        
        
# sorted color matrix
bloque0 = np.ones((100,1000,3))
bloque = np.zeros((1000,1000,3))
j = 0
for f in np.arange(n)*100:
    for k in range(3):
        block[f:f+100,:,k] = block0[:,:,k]*colors[j,k]
    j += 1
block2 = block*255
plt.imshow(block)
plt.show()

#%% save image file
a3 = block2.astype(np.uint8)
Im = Image.fromarray(a3)
Im.save("colors2.jpg")
