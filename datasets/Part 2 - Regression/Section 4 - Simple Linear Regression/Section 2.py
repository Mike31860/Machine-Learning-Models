# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:20:16 2022

@author: Migue
"""

# Plantilla de Pre Procesado
#video 1
# How can we import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

#Video 2
#Create  model of Simple Reresion with the training set
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)

#Video 3
#Predict the test set
y_prediction= regression.predict(X_test)


#video 4
#Create a Graph: Visualize the results of the training data
plt.scatter(X_train, Y_train, color="red")
plt.scatter(X_train, regression.predict(X_train), color="blue")
plt.title("Salary vs Years of experience (training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary in dollars")
plt.show


#Create a Graph: Visualize the results of the test data
plt.scatter(X_test, Y_test, color="red")
plt.scatter(X_train, regression.predict(X_train), color="blue")
plt.title("Salary vs Years of experience (training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary in dollars")
plt.show




