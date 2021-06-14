import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier


def DecisionTree(X_train , X_test , Y_train , Y_test , dataset):

	tree = DecisionTreeClassifier(random_state=0)

	tree.fit(X_train,Y_train)

	plt.figure(figsize=(8,6))
	n_features = 8
	plt.barh(range(n_features), tree.feature_importances_, align='center')
	d_features = [x for i,x in enumerate(dataset.columns) if i!=8]
	plt.yticks(np.arange(n_features), d_features)
	plt.xlabel("Feature importance")
	plt.ylabel("Feature")
	plt.ylim(-1, n_features)
	plt.show()

	print("Training accuracy using decision tree classifier : ",tree.score(X_train,Y_train) * 100)

	print("Testing accuracy using decision tree classifier : ",tree.score(X_test,Y_test) * 100)


	tree = DecisionTreeClassifier(max_depth = 3 ,random_state=0)

	tree.fit(X_train,Y_train)

	plt.figure(figsize=(8,6))
	n_features = 8
	plt.barh(range(n_features), tree.feature_importances_, align='center')
	d_features = [x for i,x in enumerate(dataset.columns) if i!=8]
	plt.yticks(np.arange(n_features), d_features)
	plt.xlabel("Feature importance")
	plt.ylabel("Feature")
	plt.ylim(-1, n_features)
	plt.show()
	

	print("Training accuracy using decision tree classifier with max depth : ",tree.score(X_train,Y_train) * 100)

	print("Testing accuracy using decision tree classifier with max depth : ",tree.score(X_test,Y_test) * 100)

def KNN(X_train , X_test , Y_train , Y_test):

	knn = KNeighborsClassifier()

	knn.fit(X_train,Y_train)

	print("Training accuracy using KNN Algorithm : ",knn.score(X_train,Y_train) * 100)

	print("Testing accuracy using KNN Algorithm  : ",knn.score(X_test,Y_test) * 100)



def RandomForest(X_train , X_test , Y_train , Y_test):

	forest = RandomForestClassifier()

	forest.fit(X_train,Y_train)

	print("Training accuracy using Random Forest Algorithm : ",forest.score(X_train,Y_train) * 100)

	print("Testing accuracy using Random Forest Algorithm  : ",forest.score(X_test,Y_test) * 100)

def Bagging(X_train , X_test , Y_train , Y_test):

	bagg = BaggingClassifier( DecisionTreeClassifier() , max_samples = 0.5 , max_features = 1.0 , n_estimators= 20)

	bagg.fit(X_train,Y_train)

	print("Training accuracy using Bagging Classifier Algorithm : ",bagg.score(X_train,Y_train) * 100)

	print("Testing accuracy using Bagging Classifier  Algorithm  : ",bagg.score(X_test,Y_test) * 100)


	
	
	

def main():

	dataset = pd.read_csv("diabetes.csv")
	print(dataset.head(3))

	features = dataset.iloc[:,:-1]
	target = dataset.iloc[: , -1]
	

	X_train , X_test , Y_train , Y_test = train_test_split(features , target , test_size = 0.2 , random_state = 66)

	DecisionTree(X_train , X_test , Y_train , Y_test , dataset)
	KNN(X_train , X_test , Y_train , Y_test)
	RandomForest(X_train , X_test , Y_train , Y_test)

	Bagging(X_train , X_test , Y_train , Y_test)

if __name__ == '__main__':
	main()