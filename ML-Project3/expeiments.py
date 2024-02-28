import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('banknote_authentication.csv')
#random_indices= {i:np.random.randint(0,len(data)) for i in range(len(data))}
random_indices=np.random.choice(len(data),size=len(data),replace=True)
print(len(random_indices))
print(len(data))
'''
count=0
for i in range(len(data)):
    for j in range(len(data)):
        if(random_indices[i]==random_indices[j] and i != j):
            #print(f'found duplicates, value is: {random_indices[i]} at indices: {i},{j}')
            #print(f'random_indices[i]={random_indices[i]},random_indices[j]={random_indices[j]}, ')
            count+=1
    random_indices = np.random.choice(len(data), size=len(data), replace=True)
print(f'total duplicates: {count}')
'''
samp_data = data.iloc[random_indices]
print(samp_data.sort_index())