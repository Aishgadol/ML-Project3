import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D

# Load the dictionary
data_dict = np.load('linreg_data_2d.npy', allow_pickle=True).item()

# Access the data as needed
x_train = data_dict['x_train']
y_train = data_dict['y_train']
x_test = data_dict['x_test']
y_test = data_dict['y_test']

#Look at the plot of the training data.
'''
plt.scatter(x_train, y_train, color='blue', s=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Generated Train')
plt.show()
'''

def kernel(xi, xj, sigma):
    return np.exp((-(np.square(np.linalg.norm(xi-xj))))/(2*np.square(sigma)))

def prepear_kernel_matrix(train, sigma):
    K = np.zeros((len(train), len(train)))
    for i in range(len(train)):
        for j in range(len(train)):
            K[i,j]=kernel(train[i],train[j],sigma)
    return K

def get_alphas(kernel, target, lamda=0.01):
    identity=np.identity(len(kernel))
    inversed=np.linalg.inv(kernel+lamda*identity)
    return (inversed @ target)

def predict(alphas, train, test, sigma):
    summy=0
    for sample_index,sample in enumerate(train):
        summy+=alphas[sample_index]*kernel(sample,test,sigma)
    return summy
