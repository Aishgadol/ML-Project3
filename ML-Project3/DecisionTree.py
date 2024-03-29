import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('banknote_authentication.csv')
# Define the ID3 decision tree class
class DecisionTree:
	def __init__(self,criterion='entropy'):
		self.tree = {}
		self.criterion=criterion

	def calculate_entropy(self, data):
		labels = data.iloc[:, -1]
		num_samples=len(data)
		unique_labels=labels.unique()
		label_counts={label:sum(labels==label) for label in unique_labels}
		entropy=0
		for label in unique_labels:
			label_prob=label_counts[label]/num_samples
			if(label_prob==0):
				entropy=entropy
			else:
				entropy -= label_prob * np.log2(label_prob)
		return entropy

	def calc_gini(self,data):
		labels=data['class']
		num_samples=len(data)
		unique_labels=labels.unique()
		label_counts={label:sum(labels==label) for label in unique_labels}
		gini=1
		for label in unique_labels:
			label_prob=label_counts[label]/num_samples
			gini -= np.square(label_prob)

		return gini

	def calculate_information_gain(self, data, feature):
		if(self.criterion == 'gini'):
			total_entropy=self.calc_gini(data)
		else:
			total_entropy = self.calculate_entropy(data)
		information_gain = total_entropy
		data_at_feature_numpy_sorted=np.sort(data[feature].to_numpy())
		#i choose 10 evenly spaced thresholds between the max and min in the feature
		'''
		value_indices = np.linspace(0, len(data)-1, num=102,endpoint=True,dtype=int)[1:-1]  # Exclude the max and min values
		values=data_at_feature_numpy_sorted[value_indices]
		'''
		values=np.linspace(np.min(data_at_feature_numpy_sorted),max(data_at_feature_numpy_sorted),num=52,endpoint=True)[1:-1]
		best_treshold = None
		best_gain = 0
		for value in values:
			data_above_threshold=self.filter_data(data,feature,value,False)
			data_below_threshold=self.filter_data(data,feature,value)
			tot_samples=len(data)
			if(self.criterion == 'gini'):
				gain_feature=total_entropy-((len(data_above_threshold)/tot_samples)*self.calc_gini(data_above_threshold))-((len(data_below_threshold)/tot_samples)*self.calc_gini(data_below_threshold))
			else:
				gain_feature=total_entropy-((len(data_above_threshold)/tot_samples)*self.calculate_entropy(data_above_threshold))-((len(data_below_threshold)/tot_samples)*self.calculate_entropy(data_below_threshold))
			if(gain_feature>=best_gain):
				best_gain=gain_feature
				best_treshold=value
		return best_gain, best_treshold

	def filter_data(self, data, feature, value, left=True):
		if left:
			return data[data[feature] <= value].drop(feature, axis=1)
		else:
			return data[data[feature] > value].drop(feature, axis=1)

	def create_tree(self, data, depth=0):
		# Recursive function to create the decision tree
		labels = data.iloc[:, -1]

		# Base case: if all labels are the same, return the label
		if len(np.unique(labels)) == 1:
			return list(labels)[0]

		features = data.columns.tolist()[:-1]
		# Base case: if there are no features left to split on, return the majority label
		if len(features) == 0:
			unique_labels, label_counts = np.unique(labels, return_counts=True)
			majority_label = unique_labels[label_counts.argmax()]
			return majority_label

		selected_feature = None
		best_gain = 0
		best_treshold = None

		for feature in features:
			gain, treshold = self.calculate_information_gain(data, feature)
			if gain >= best_gain:
				selected_feature = feature
				best_treshold = treshold
				best_gain = gain

		# Create the tree node
		tree_node = {}
		tree_node[(selected_feature, f"<={best_treshold}")] = self.create_tree(self.filter_data(data, selected_feature, best_treshold, left=True), depth+1)
		tree_node[(selected_feature, f">{best_treshold}")] = self.create_tree(self.filter_data(data, selected_feature, best_treshold, left=False), depth+1)

		# check if can unite them.
		if not isinstance(tree_node[(selected_feature, f"<={best_treshold}")], dict) and \
				not isinstance(tree_node[(selected_feature, f">{best_treshold}")], dict):
			if tree_node[(selected_feature, f"<={best_treshold}")] == tree_node[(selected_feature, f">{best_treshold}")]:
				return tree_node[(selected_feature, f"<={best_treshold}")]

		return tree_node

	def fit(self, data):
		self.tree = self.create_tree(data)

	def predict(self, X):
		X = [row[1] for row in X.iterrows()]

		# Predict the labels for new data points
		predictions = []

		for row in X:
			current_node = self.tree
			while isinstance(current_node, dict):
				split_condition = next(iter(current_node))
				feature, value = split_condition
				treshold = float(value[2:])
				if row[feature] <= treshold:
					current_node = current_node[feature, f"<={treshold}"]
				else:
					current_node = current_node[feature, f">{treshold}"]
			predictions.append(current_node)

		return predictions

	def _plot(self, tree, indent):
		depth = 1
		for key, value in tree.items():
			if isinstance(value, dict):
				print(" " * indent + str(key) + ":")
				depth = max(depth, 1 + self._plot(value, indent + 2))
			else:
				print(" " * indent + str(key) + ": " + str(value))
		return depth

	def plot(self):
		depth = self._plot(self.tree, 0)
		print(f'depth is {depth}')

decision_tree_model=DecisionTree()
decision_tree_model.fit(data)
decision_tree_model.plot()

train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['class'])

print(f'100 thresholds')
for criterion in ["entropy", "gini"]:
  print(f"----------- {criterion} -----------")
  decision_tree_model=DecisionTree(criterion)
  decision_tree_model.fit(train)
  preds=decision_tree_model.predict(train.drop('class',axis=1))
  acc=sum(predict==label for predict,label in zip(preds,train['class']))/len(train)
  print(f'Training accuracy is {acc}')
  preds=decision_tree_model.predict(test.drop('class',axis=1))
  acc=sum(predict==label for predict,label in zip(preds,test['class']))/len(test)
  print(f'Test accuracy is {acc}')


