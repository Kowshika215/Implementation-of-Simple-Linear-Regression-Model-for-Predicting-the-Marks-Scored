# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1..Import necessary libraries

2..Load dataset

3.. Define input (independent variable) and output (dependent variable)

4..Split dataset into training and testing sets

5.Build Linear Regression model

6.Predict test set results

7.Visualize Training set results

8.Visualize Testing set results

9.Evaluate model performance


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Kowshika.R
RegisterNumber: 212224220049 
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head()
df.tail()
print(df.head())
print(df.tail())
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="brown")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae )
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:

To Read Head and Tail Files

<img width="223" height="255" alt="Screenshot 2025-08-28 094817" src="https://github.com/user-attachments/assets/3f20d1df-aca9-4eca-83f4-b687a64b2c83" />

Compare Dataset

<img width="733" height="577" alt="Screenshot 2025-08-28 094829" src="https://github.com/user-attachments/assets/991c89ad-4f97-4a8b-80c3-c9843cd325f2" />

Predicted Value

<img width="728" height="86" alt="Screenshot 2025-08-28 094837" src="https://github.com/user-attachments/assets/d83316b1-4c17-403d-ac8e-a13799cf25f5" />

Graph for Training Set

<img width="814" height="577" alt="Screenshot 2025-08-28 094845" src="https://github.com/user-attachments/assets/9dcbed39-faa9-4567-bfd9-073daf4ab0b0" />

Graph for Testing Set

<img width="784" height="665" alt="Screenshot 2025-08-28 094853" src="https://github.com/user-attachments/assets/7ba035d8-141c-4fa2-b851-de800b259d2e" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
