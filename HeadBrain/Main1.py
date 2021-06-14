import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
 


def HeadBrain(file_name):

	dataset = pd.read_csv(file_name)
	print("Size of our dataset is",dataset.shape)

	X =	dataset["Head Size(cm^3)"].values 

	Y = dataset["Brain Weight(grams)"].values

	X=X.reshape((-1,1))


	obj = LinearRegression()

	obj.fit(X,Y)

	result = obj.predict(X)

	RMS = obj.score(X,Y)

	print(RMS)

def main():

	file_name = input("Enter the file name of the dataset\n")

	HeadBrain(file_name)

	
if __name__ == '__main__':
	main()