import numpy as np

import pandas as pd

import seaborn as sn

from seaborn import countplot

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure,show

from sklearn.model_selection import train_test_split
	
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix


def TitanicLogistic():

#-----------------Step 1 - Load Data----------------------------------------#
	
	Titanic_Data = pd.read_csv("Titanic.csv")
	print("First five records of dataset:")
	print(Titanic_Data.head())
	print("Total Number of records are:",len(Titanic_Data))

# #-----------------Step 2 - Visualize the Data----------------------------------------#
	
# 	print("Visualization of Survived and non Survived Passengers")

# 	figure()
# 	countplot( data=Titanic_Data , x = "Survived" ).set_title("Survived vs Non Survived Ratio")
# 	show()

# 	print("Visualization of Gender")
# 	figure()
# 	countplot( data=Titanic_Data , x ="Survived", hue = "Sex").set_title("Ratio of Survived and Non Survived Male Female")
# 	show()


# 	print("Visualization of Passenger Class(Pclass)")
# 	figure()
# 	countplot( data=Titanic_Data , x ="Survived" , hue = "Pclass").set_title("Survived Non Survived Classes")
# 	show()

# 	print("Visualization of Survived and Non Survived according to Age")
# 	figure()
# 	Titanic_Data["Age"].plot.hist().set_title("Age of Passengers")
# 	show()

#-----------------Step 3- CLean the Data  (Data Rangling)----------------------------------------#

	Titanic_Data.drop( "zero" , axis = 1 , inplace = True )
	print("Data after column removal")
	print(Titanic_Data.head())

	Titanic_Data.drop( "Age" , axis = 1 , inplace = True )
	print("Data after column removal")
	print(Titanic_Data.head())

	Sex = pd.get_dummies( Titanic_Data["Sex"] )
	print(Sex.head())
	Sex = pd.get_dummies( Titanic_Data["Sex"] , drop_first=True )
	print("Sex column after updation")
	print(Sex.head())

	Pclass = pd.get_dummies( Titanic_Data["Pclass"])
	print(Pclass.head(3))
	Pclass = pd.get_dummies( Titanic_Data["Pclass"] , drop_first=True )
	print("Pclass column after updation")
	print(Pclass.head())

	#Concat the two irrelevant columns from the dataset
	Titanic_Data = pd.concat( [ Titanic_Data , Sex , Pclass ] , axis = 1 )
	print(Titanic_Data.head(3))

	print("Removing irrelevant fields from dataset")

	# Titanic_Data.drop( "Sex" , axis = 1 , inplace = True )
	# Titanic_Data.drop( "Pclass" , axis = 1 , inplace = True )
	# Titanic_Data.drop( "sibsp" , axis = 1 , inplace = True )
	# Titanic_Data.drop( "Embarked" , axis = 1 , inplace = True )
	# Titanic_Data.drop( "Parch" , axis = 1 , inplace = True )

	Titanic_Data.drop(["Sex", "sibsp" , "Parch" , "Pclass" , "Embarked"] , axis = 1 , inplace = True) # ----same output for above 81 to 85th line

	print(Titanic_Data.head(3))




#-----------------Step 4- Split the Data----------------------------------------#
	
	X = Titanic_Data.drop("Survived" , axis= 1 )
	Y = Titanic_Data["Survived"]                           # We are not using 'inplace = True' so sequence doesnt matters

	X_train , X_test , Y_train , Y_test = train_test_split( X , Y , test_size = 0.5 )

#-----------------Step 5- Train and Test the Data----------------------------------------#	

	obj = LogisticRegression()      # or use LogisticRegression(max_itr = 2000)

	obj.fit( X_train , Y_train )

	output = obj.predict( X_test )

	accuracy = accuracy_score(output , Y_test )
	print("Accuracy is :",accuracy*100,"%")

	print("Confusion Matrix is")
	print(confusion_matrix(Y_test , output ))


	
def main():
	TitanicLogistic()


if __name__ == '__main__':
	main()