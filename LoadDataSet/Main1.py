from MyModules import *



def KNN():

	url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"

	dataset =pd.read_csv(url)

	df=pd.DataFrame(dataset)
	print(dataset)

	data = df.iloc[: , :-1]

	target = df.iloc[: ,-1]


	data_train , data_test ,target_train , target_test = train_test_split(data , target , test_size = 0.5)

	
	cobj = KNeighborsClassifier()

	cobj.fit(data_train,target_train)

	output = cobj.predict(data_test)

	Accuracy = accuracy_score(target_test , output)

	print("Accuracy of KNN Algorithm:",Accuracy*100,"%")


def main():

	KNN()

if __name__ == '__main__':
	main()
