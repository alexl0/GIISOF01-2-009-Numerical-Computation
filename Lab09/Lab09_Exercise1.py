# -*- coding: utf-8 -*-
"""
k-means
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
#---------------------- Generate data
np.random.seed(7)

x1 = np.random.standard_normal((100,2))*0.6+np.ones((100,2))
x2 = np.random.standard_normal((100,2))*0.5-np.ones((100,2))
x3 = np.random.standard_normal((100,2))*0.4-2*np.ones((100,2))+5
X = np.concatenate((x1,x2,x3),axis=0)

# plot
plt.plot(X[:,0],X[:,1],'k.')
plt.show()

k = 3 # number of centroids

#---------------------- Initialize centroids
m, n = X.shape
mi = np.min(X,axis=0)
ma = np.max(X,axis=0)

centroids = np.random.rand(k,n) # in [0,1]
centroids = (ma-mi)*centroids + mi # in [mi,ma]

# plot
plt.plot(X[:,0],X[:,1],'k.')
plt.plot(centroids[:,0],centroids[:,1],'ro',markersize=10)
plt.show()

#---------------------- Initialize labels

labels = np.zeros(m)

#----------------------  Iteratively:
#----------------------  (a) Assign centroids
#----------------------  (b) Find new centroids
d = np.zeros(k)
for iteration in range(5):
    print('----------  Iteration '+str(iteration))
    # (a) Assign centroids
    for i in range(m):
        for j in range(k):
            d[j] = norm(X[i,:]-centroids[j,:])
        labels[i] = np.argmin(d)
    
    # plot
    for j in range(k):    
        plt.plot(X[labels==j,0],X[labels==j,1],'.')
    plt.plot(centroids[:,0],centroids[:,1],'o',markersize=10)
    plt.title('Assign the points to the nearest centroid')
    plt.show() 
    
    # (b) Find new centroids
    for j in range(k):
        centroids[j] = np.mean(X[labels==j,:])
 
    # plot
    for j in range(k):    
        plt.plot(X[labels==j,0],X[labels==j,1],'.')
    plt.plot(centroids[:,0],centroids[:,1],'o',markersize=10)
    plt.title('Reallocate centroids')
    plt.show()
            
