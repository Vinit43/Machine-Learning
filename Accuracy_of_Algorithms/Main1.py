from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier


def Decision(data_train,data_test,target_train,target_test):

	


	cobj = tree.DecisionTreeClassifier()

	cobj.fit(data_train,target_train)

	output = cobj.predict(data_test)

	Accuracy = accuracy_score(target_test , output)

	print("Accuracy of Decision Tree Algorithm:",Accuracy*100,"%")



def KNN(data_train,data_test,target_train,target_test):

	


	cobj = KNeighborsClassifier()

	cobj.fit(data_train,target_train)

	output = cobj.predict(data_test)

	Accuracy = accuracy_score(target_test , output)

	print("Accuracy of KNN Algorithm:",Accuracy*100,"%")

def main():

	dataset = load_iris()

	data= dataset.data
	target=dataset.target

	data_train,data_test,target_train,target_test = train_test_split(data,target,test_size = 0.5)


	Decision(data_train,data_test,target_train,target_test)
	KNN(data_train,data_test,target_train,target_test)

if __name__ == '__main__':
	main()