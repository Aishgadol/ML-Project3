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
		values=np.linspace(np.min(data_at_feature_numpy_sorted),max(data_at_feature_numpy_sorted),num=12,endpoint=True)[1:-1]
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

# Define the ID3 decision tree class
class RandomForest:
	def __init__(self, n_estimators=3, method='simple', criterion='entropy'):
		self.forest = []
		self.criterion = criterion
		self.n_estimators = n_estimators
		self.method = method
		self.tree_count=0

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
			self.tree_count+=1
			#print(f'creating tree number{self.tree_count}')
			samp_data = data.iloc[self.sample_data(data)]
			# Implement here
			DTmodel=DecisionTree(criterion=self.criterion)
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
'''
dict1 = {'entropy': [], 'gini': []}
criterions = ['entropy', 'gini']
for num_estimators in range(1,5):
	dict1 = {'entropy': [], 'gini': []}
	for crt in criterions:

		forest = RandomForest(n_estimators=num_estimators, method='simple', criterion=crt)
		forest.fit(train)

		acc = forest.score(train)
		dict1[crt].append(acc)

		acc = forest.score(test)
		dict1[crt].append(acc)

	print(f'using {num_estimators} estimators:')
	df = pd.DataFrame(dict1, columns=criterions, index=['train', 'test'])
	print(df)
'''

'''
def getAccuracyUsingRandomForest(train,test):
    accs={'entropy': [], 'gini': []}
    criterions = ['entropy', 'gini']
    for crt in criterions:
        forest = RandomForest(n_estimators=3, method='simple', criterion=crt)
        forest.fit(train)
        acc = forest.score(train)
        accs[crt].append(acc)
        acc = forest.score(test)
        accs[crt].append(acc)
    return accs
'''
def KFold2(data, model, cv=5):
    kf = KFold(n_splits=cv)
    scores = []

    for train_index, test_index in kf.split(data):
        this_train = data.iloc[train_index]
        this_test = data.iloc[test_index]
        model.fit(this_train)
        scores.append(model.score(this_test))
    return np.mean(scores)

correct_entropy = []
correct_gini = []
best_num_estimators=0
best_gini_acc=0
best_entropy_acc=0
for i in tqdm(range(3,11)):
	forest = RandomForest(n_estimators=i, method='simple', criterion='gini')
	current_gini_acc=KFold2(data=train, model=forest, cv=5)
	correct_gini.append(current_gini_acc)
	forest = RandomForest(n_estimators=i, method='simple', criterion='entropy')
	current_entropy_acc=KFold2(data=train, model=forest, cv=5)
	correct_entropy.append(current_entropy_acc)
	if (best_gini_acc < current_gini_acc and best_entropy_acc<current_entropy_acc):
		best_entropy_acc=current_entropy_acc
		best_gini_acc=current_gini_acc
		best_num_estimators=i

plt.plot(range(3,11), np.array(correct_entropy), label='entropy')
plt.plot(range(3,11), np.array(correct_gini), label='gini')
plt.title(f'entropy vs gini, best num estimators: {best_num_estimators}, 10 thresholds')
plt.legend(loc='upper left')
plt.xlabel('trees num')
plt.ylabel('avg accuracy')
plt.show()