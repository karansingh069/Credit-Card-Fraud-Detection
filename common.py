import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import gridspec 

data = pd.read_csv("creditcard.csv")  
data.head() 


print(data.shape) 
print(data.describe()) 


fraud = data[data['Class'] == 1] 
valid = data[data['Class'] == 0] 
outlierFraction = len(fraud)/float(len(valid)) 
print(outlierFraction) 
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1]))) 
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0]))) 


X = data.drop(['Class'], axis = 1) 
Y = data["Class"] 
print(X.shape) 
print(Y.shape) 
 
xData = X.values 
yData = Y.values 



from sklearn.model_selection import train_test_split 
xTrain, xTest, yTrain, yTest = train_test_split( 
     xData, yData, test_size = 0.2, random_state = 42) 
