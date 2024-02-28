import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from DecisionTree import DecisionTree
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from sklearn.model_selection import KFold

data = pd.read_csv('banknote_authentication.csv')
train, test = train_test_split(data, test_size=0.2, stratify=data['class'], random_state=42)

# Define the ID3 decision tree class
class RandomForest:
	def __init__(self, n_estimators=3, method='simple', criterion='entropy'):
		self.forest = []
		self.criterion = criterion
		self.n_estimators = n_estimators
		self.method = method

	def select_features(self, data):
		np.random.seed(40+len(self.forest))

		if self.method == 'sqrt':
			m = int(np.sqrt(len(data.columns)-1))
		elif self.method == 'log':
			m = int(np.log2(len(data.columns)-1))
		else:
			m = np.random.randint(0, len(data.columns))

		incidies = np.random.choice(np.arange(0, len(data.columns)-1), size=m, replace=False)
		features = list(data.columns[incidies])
		return data[features + ['class']]

	def sample_data(self, data):
		# This method samples len(data) with repitition from data.
		# You can use numpy to select random incidies.
		return np.random.choice(len(data), size=len(data), replace=True)

	def fit(self, data):
		self.forest = []
		for _ in range(self.n_estimators):
			samp_data = data.iloc[self.sample_data(data)]
			# Implement here
			DTmodel=DecisionTree()
			samp_data=self.select_features(samp_data)
			DTmodel.fit(samp_data)
			self.forest.append(DTmodel)

	def _predict(self, X):
		# Predict the labels for new data points
		predictions = []

		preds = [tree.predict(X) for tree in self.forest]
		preds = list(zip(*preds))
		predictions = [Counter(est).most_common(1)[0][0] for est in preds]

		return predictions

	def score(self, X):
		pred = self._predict(X)
		return (pred == X.iloc[:,-1]).sum() / len(X)