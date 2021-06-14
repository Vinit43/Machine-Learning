import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def MeanData(arr):

    size = len(arr)
    sum = 0

    for i in range(size):

        sum = sum+arr[i]

    return (sum/size)




def HeadBrain():


    dataset = pd.read_csv("HeadBrain.csv")

    print("Size is:",dataset.shape)

    X = dataset["Head Size(cm^3)"].values

    Y = dataset["Brain Weight(grams)"].values

    print("Lenght os x is:",len(X))

    print("Lenght os x is:",len(Y))

    X_mean = MeanData(X)

    Y_mean = MeanData(Y)

    print("Mean of X is:",X_mean)

    print("Mean of Y is:",Y_mean)

    Numerator = 0

    Denominator = 0

    for i in range(len(X)):

        Numerator = Numerator + ((X[i] - X_mean)(Y[i] - Y_mean)) 
        Denominator = Denominator + ((X[i] - X_mean)**2)

    m = Numerator/Denominator

    print("Value of Slope(m) is :",m)

    c = Y_mean - m(X_mean)

    print("Value of y intercept(c) is:",c)

    X_Start = np.min(X)

    X_End = np.max(X)

    x = np.linspace(X_Start,X_End)


    y = m*x + c

    plt.plot(x , y , color = 'r' , label = "Line of Regression")

    plt.scatter(X , Y , color = 'b', label = "Data Plot")


    plt.xlabel("Head Size(cm)")
    plt.ylabel("Brain Weigth(gm)")

    plt.legend()
    plt.show()



def main():

    #file_name = input("Enter the name of file of dataset")


    HeadBrain()

if __name__ == '__main__':
	main()









# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt


# def MeanData(arr):

# 	size = len(arr)
# 	sum=0

# 	for i in range(size):
# 		sum=sum+arr[i]

# 	return sum/size


# def HeadBrain(file_name):

# 	dataset = pd.read_csv(file_name)
# 	print("Size of our dataset is",dataset.shape)

# 	X =	dataset["Head Size(cm^3)"].values 

# 	Y = dataset["Brain Weight(grams)"].values

# 	print("Lenght of X",len(X))
# 	print("Lenght of Y",len(Y))

# 	X_Mean = MeanData(X)
# 	Y_Mean = MeanData(Y)

# 	print("X mean is ",X_Mean)
# 	print("Y mean is ",Y_Mean)

# 	numerator = 0
# 	denominator = 0

# 	for i in range(len(X)):

# 		numerator = numerator + ((X[i] - X_Mean)*(Y[i]-Y_Mean))
# 		denominator = denominator + ((X[i] - X_Mean)**2)

# 	m = numerator/denominator

# 	print("Slope is i.e Value of m is:",m)

# 	c = Y_Mean - (m*(X_Mean)) 

# 	print("Y intercept i.e value of c is:",c)

# 	X_start = np.min(X)
# 	X_end = np.max(X)

# 	x = np.linspace(c,X_end)

# 	y= m*x + c
	

# 	plt.plot(x,y,color='r',label="Line of Regression")
# 	plt.scatter(X,Y,color='b',label="Data points")

# 	plt.xlabel("Head Size (cm)")
# 	plt.ylabel("Brain Weight (gm)")

# 	plt.legend()
	
# 	plt.show()


# 	Numerator = 0
# 	Denominator = 0

# 	for i in range(len(X)):
# 		Numerator = Numerator+ (((m*X[i] + c) - Y_Mean)**2)
# 		Denominator = Denominator + ((Y[i]-Y_Mean)**2)

# 	rSquare = Numerator/Denominator
# 	print("R Square is :",rSquare)

# def main():

# 	file_name = input("Enter the file name of the dataset\n")

# 	HeadBrain(file_name)

	
# if __name__ == '__main__':
# 	main()