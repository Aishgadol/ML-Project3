import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from sklearn.model_selection import KFold

data = pd.read_csv('banknote_authentication.csv')
train, test = train_test_split(data, test_size=0.2, stratify=data['class'], random_state=42)
