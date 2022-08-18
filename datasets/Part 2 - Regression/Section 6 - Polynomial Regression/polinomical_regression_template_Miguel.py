# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:14:33 2022

@author: Migue
"""



# Regresión polinómica

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Adjust the regretion with the dataset
#Create our model of regression. We can apply any model we would like to work with.


#Predict of our models
lin_regression.predict([[6.5]])
lin_regression_2.predict(poly_regression.fit_transform([[6.5]]))



#Visualize the results of the Polinomical model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, Y, color ="red")
plt.plot(X_grid, lin_regression_2.predict(poly_regression.fit_transform(X_grid)), color ="blue")
plt.title("Polinomial Regression Model")
plt.xlabel("Position of the Employee")
plt.ylabel("Salary in $")
plt.show()


