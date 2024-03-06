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

#using sigma=4, plotting line:
xx = np.arange(0, 100, 0.1).reshape((1000,1))
yy = []
train_kernel = prepear_kernel_matrix(x_train, sigma=4)
alphas = get_alphas(train_kernel, y_train)

for samp in xx:
  yy.append(predict(alphas, x_train, samp, sigma=4))

plt.scatter(x_train, y_train, color='blue', s=2, label='train')
plt.scatter(x_test, y_test, color='red', s=2, label='test')
plt.plot(xx, yy, color='black')
plt.show()

#printing MSE of train and test given sigma=4
mse = 0
for idx, samp in enumerate(x_train):
  mse += (predict(alphas, x_train, samp, sigma=4) - y_train[idx]) ** 2
mse = mse / len(x_train)
print(f'train mse is {mse}')

mse = 0
for idx, samp in enumerate(x_test):
  mse += (predict(alphas, x_train, samp, sigma=4) - y_test[idx]) ** 2
mse = mse / len(x_test)
print(f'test mse is {mse}')

#creating validation set with 20 samples and using train_test_split to tune better sigma and lamda
def getBestSigmaLamda(x_train,y_train):
    X_train_for_val,X_val, y_train_for_val,y_val=train_test_split(x_train,y_train, test_size=20, random_state=42)
    sigmas=range(20,50)
    lamdas=np.arange(0.01,0.3,0.01)
    min_error=1
    best_lamda=0
    best_sigma=0
    mse=0
    best_mse=1
    for lamda in lamdas:
        for sigma in sigmas:
            print(f'lamda:{lamda}, sigma:{sigma}')
            mse=0
            curr_train_kernel = prepear_kernel_matrix(X_train_for_val, sigma=sigma)
            curr_alphas = get_alphas(curr_train_kernel, y_train_for_val,lamda=lamda)
            for idx, samp in enumerate(X_train_for_val):
                mse += (predict(curr_alphas, X_train_for_val, samp, sigma=4) - y_train_for_val[idx]) ** 2
            mse = mse / len(X_train_for_val)
            print(f'train mse is {mse}')
            mse=0
            for idx,samp in enumerate(X_val):
                mse+=(predict(curr_alphas,X_train_for_val,samp,sigma=sigma)-y_val[idx])**2
            mse = mse / len(X_val)
            if(mse<best_mse):
                best_mse=mse
                best_lamda=lamda
                best_sigma=sigma
            print(f'val mse is {mse}\n')
    print(f'best mse: {best_mse}, best sigma: {best_sigma}, best lamda: {best_lamda}')
    return best_sigma,best_lamda
best_sigma,best_lamda=getBestSigmaLamda(x_train,y_train)

#training the model with new tuned hyperparameters:
train_kernel = prepear_kernel_matrix(x_train, sigma=best_sigma)
alphas = get_alphas(train_kernel, y_train,best_lamda)
mse = 0
for idx, samp in enumerate(x_train):
  mse += (predict(alphas, x_train, samp, sigma=best_sigma) - y_train[idx]) ** 2
mse = mse / len(x_train)
print(f'train mse is {mse}')

mse = 0
for idx, samp in enumerate(x_test):
  mse += (predict(alphas, x_train, samp, sigma=best_sigma) - y_test[idx]) ** 2
mse = mse / len(x_test)
print(f'test mse is {mse}')

#plotting old vs new regressor, line in green in new regression line using new hyperparameters
xx = np.arange(0, 100, 0.1).reshape((1000,1))
yy2 = []

for samp in xx:
  yy2.append(predict(alphas, x_train, samp, sigma=best_sigma))

plt.scatter(x_train, y_train, color='blue', s=2, label='train')
plt.scatter(x_test, y_test, color='red', s=2, label='test')
plt.plot(xx, yy, color='black', label='origin plot')
plt.plot(xx, yy2, color='green', label='tuned plot')
plt.legend()
plt.show()
