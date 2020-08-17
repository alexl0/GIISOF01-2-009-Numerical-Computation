# -*- coding: utf-8 -*-
"""
che-guevara
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

I = Image.open("che-guevara.jpg")

# convert to black and white
I1 = I.convert('L')

# to numpy array
a = np.asarray(I1,dtype=np.float32)
plt.imshow(a,cmap='gray')
plt.axis('off')
plt.show()


# crop
a1 = a[50:530,:]
plt.imshow(a1,cmap='gray')
plt.axis('off')
plt.show()


# rearrange and cluster
x, y = a1.shape
a2 = a1.reshape(-1,1)

k_means = KMeans(n_clusters=2) 
k_means.fit(a2) 

centroids = k_means.cluster_centers_.squeeze()
labels = k_means.labels_


# reshape and show result
a3 = centroids[labels]
a4 = a3.reshape(x,y)

plt.imshow(a4,cmap='gray')
plt.axis('off')
plt.show()

# save as image file
a5 = (a4 - np.min(a4))/(np.max(a4) - np.min(a4))*255
a6 = a5.astype('uint8')
I2 = Image.fromarray(a6)
I2.save('poster.jpg')
