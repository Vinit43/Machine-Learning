
from sklearn.datasets import load_iris


import numpy as np

from sklearn import tree


def main():

	dataset = load_iris()


	print("Features of dataset\n",dataset.feature_names,"\nTargets are:\n",dataset.target_names)

	Index = [1,51,101]

	test_target = dataset.target[Index]
	test_feature = dataset.data[Index]

	train_target = np.delete(dataset.target,Index)
	train_feature = np.delete(dataset.data,Index,axis=0)

	obj = tree.DecisionTreeClassifier()

	obj.fit(train_feature,train_target)

	result = obj.predict(test_feature)

	print("Result prediction by ML",result)
	print("Result expected",test_target)




if __name__ == '__main__':
	main()