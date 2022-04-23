# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
STEP1:
Import the standard Libraries.

STEP2:
Set variables for assigning dataset values.

STEP3:
Import linear regression from sklearn.

STEP4:
Assign the points for representing in the graph

STEP5:
Predict the regression for marks by using the representation of the graph.

STEP6:
Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Pavan kishore.M
RegisterNumber:212221230076  
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('student.csv')
dataset.head()
X = dataset.iloc[:,:-1].values
X
Y = dataset.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue") 
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
![ml1](https://user-images.githubusercontent.com/94154941/164909296-88329a73-d268-4505-83b5-002290157131.png)


![ml2](https://user-images.githubusercontent.com/94154941/164909298-9c61dad7-544f-492d-b078-33868196e91d.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
