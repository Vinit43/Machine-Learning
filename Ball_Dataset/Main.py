
# This is supervised Machine Learning....
#Every supervized ML goes with Labeled data


from sklearn import tree       #sklearn is scikit-learn library

def ML(weight,surface):

	Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
	Labels = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

	dobj = tree.DecisionTreeClassifier()

	dobj.fit(Features,Labels)


	result = dobj.predict([[weight,surface]])
	if result==1:

		print("Ball is Tennis")
	else:
		print("Ball is Cricket")

def main():

	weight = int(input("Enter the weight\n"))
	surface =input("Enter the surface\n")

	if surface.lower() =="rough":
		surface = 1
	elif surface.lower()=="smooth":
		surface = 0
	else:
		print("Invalid input")

	ML(weight,surface)

	
if __name__ == '__main__':
	main()