# -*- coding: utf-8 -*-
"""
ELM
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#%%
# data files are in the folder "data" that is in the working directory
data_train = np.loadtxt('./data/data_train1.txt')
labels_train = np.loadtxt('./data/labels_train1.txt')

data_test = np.loadtxt('./data/data_test1.txt')
labels_test = np.loadtxt('./data/labels_test1.txt')
#%%  Some parameters                
number_of_classes= 10        # number of classes of digits

n = data_train.shape[0]      # number of rows (images) in training matrix
m = data_test.shape[0]       # number of rows (images) in testing matrix
I = np.identity(n)           # n x n identity matrix
#%% Normalize data (data to)
H = data_train/255.
H_test = data_test/255.
#%%
Y = np.zeros((n, number_of_classes))
for i in range(n):
    Y[i, int(labels_train[i])] = 1
#%%    
def KernelRBF(X,Y,sigma):
    m = X.shape[0]
    n = Y.shape[0]
    K = np.zeros((m,n))
    
    for i in range(m):
        for j in range(n):
            dif = np.linalg.norm(X[i,:]-Y[j,:])
            K[i,j] = np.exp(-dif**2/sigma)
            
    return K
#%% Find best parameters in a grid
C_list = [ 1., 10., 100., 1000.]
sigma_list = [ 1., 10., 100., 1000.]
best_result = 0.
for C in C_list:
    for sigma in sigma_list:
        print(' C = ', C , 'sigma = ',sigma)
        
        # Build model
        OmegaP = KernelRBF(H, H, sigma)  
        W = np.linalg.solve(I/C + OmegaP, Y)
        
        # Predict
        OmegaP_test = KernelRBF(H_test, H, sigma)
        YP_test = np.dot(OmegaP_test, W)
        
        # prediction
        predictedP_test = YP_test.argmax(axis=1)
        
        # success percentage
        percent = np.sum(predictedP_test == labels_test)/float(m)*100.
        print('Testing success = %.1f%%' % percent)
        
        # store best result
        if percent > best_result:
            best_C = C
            best_sigma = sigma
            best_result = percent
            best_prediction = predictedP_test
#%% Plot results with best parameters
print('\nBest C      = ',best_C)   
print('Best sigma  = ',best_sigma)   
print('Best result = %5.1f%%' % best_result)          
# confusion matrix
mc = confusion_matrix(labels_test, best_prediction)

print(u'\nConfusion matrix\n')
print (mc)

plt.figure(figsize=(6,6))
ticks = range(10)
plt.xticks(ticks)
plt.yticks(ticks)
plt.imshow(mc,cmap=plt.cm.Blues,interpolation='nearest')
plt.colorbar(shrink=0.8)
w, h = mc.shape
for i in range(w):
    for j in range(h):
        plt.annotate(str(mc[i][j]), xy=(j, i), 
                    horizontalalignment='center',
                    verticalalignment='center')
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.title('Confusion matrix')
plt.show()

# Plot labels and images
print('\r')
print('Labels predicted for testing samples')
for i in range(0,m):
    if i%10 == 9:
        print(best_prediction[i]),
    else:
        print(best_prediction[i], end=" "),

print('\n')
print('Imágenes que corresponden a las etiquetas anteriores')
plt.figure(figsize=(8,8))
for k in range(0, 100):
    plt.subplot(10, 10, k+1)
    image = data_test[k, ]
    image = image.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
plt.show()
