# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:20:21 2022

@author: Migue
"""

#Polinomical regretion

# Plantilla de Pre Procesado Datos Categoricos

# How can we import the libraries
import numpy as np
# libreria para representacion grafica
import matplotlib.pyplot as plt 
import pandas as pd

#Import the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

# Adjust the lineal regretion with the dataset

from sklearn.linear_model import  LinearRegression 
lin_regression=LinearRegression();
lin_regression.fit(X, Y)



#Adjust the polinomical regretion with the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regression=PolynomialFeatures(degree=6)
X_poly=poly_regression.fit_transform(X)
lin_regression_2=LinearRegression()
lin_regression_2.fit(X_poly, Y)

#Visualize the results of the lineal model
plt.scatter(X, Y, color ="red")
plt.plot(X, lin_regression.predict(X), color ="blue")
plt.title("Linear Regression Model")
plt.xlabel("Position of the Employee")
plt.ylabel("Salary in $")
plt.show()

#Visualize the results of the Polinomical model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, Y, color ="red")
plt.plot(X_grid, lin_regression_2.predict(poly_regression.fit_transform(X_grid)), color ="blue")
plt.title("Polinomial Regression Model")
plt.xlabel("Position of the Employee")
plt.ylabel("Salary in $")
plt.show()

#Predict of our models
lin_regression.predict([[6.5]])
lin_regression_2.predict(poly_regression.fit_transform([[6.5]]))












