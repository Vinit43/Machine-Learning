import pandas as pd

def main():

	df = pd.read_csv('iris1.csv')
	print(df.head())

if __name__ == '__main__':
	main()