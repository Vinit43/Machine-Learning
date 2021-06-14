import pandas as pd

from sklearn.datasets import load_wine

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

def DecisionTree(data_train , data_test , target_train , target_test):

	dobj = tree.DecisionTreeClassifier()

	dobj.fit(data_train , target_train)

	# result = dobj.predict(data_test)

	# accuracy = accuracy_score(target_test , result)

	# print("accuracy is:",accuracy*100,"%")

	output = dobj.predict([[14.23,1.71,2.43,15.6,127,2.8,3.06,.28,2.29,5.64,1.04,3.92,1065]])

	print("THE OUTPUT PREDCITED IS:",output)


def main():

	dataset = load_wine()

	data = dataset.data

	print(data)

	target = dataset.target

	data_train , data_test , target_train , target_test  = train_test_split(data , target , test_size = 0.2)

	DecisionTree(data_train , data_test , target_train , target_test)

if __name__ == '__main__':
	main()