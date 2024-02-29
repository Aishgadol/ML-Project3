import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('banknote_authentication.csv')
#random_indices= {i:np.random.randint(0,len(data)) for i in range(len(data))}
random_indices=np.random.choice(len(data),size=len(data),replace=True)

duck={'duck':[3,4], 'goose':[7,20]}
print({crt:np.mean(duck[crt]) for crt in duck})