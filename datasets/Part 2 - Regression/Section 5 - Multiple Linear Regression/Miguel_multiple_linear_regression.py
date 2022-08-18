# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 13:01:12 2022

@author: Migue
"""

#Regresion Lineal multiple
# Plantilla de Pre Procesado

# How can we import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset
dataset = pd.read_csv("50_startups.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#Codificar datos categoricos
# There are two different categories: ordinal and categorica, it is ordinal when it has an order
#Category if a Colum that does not represent a number, it represents a label for the users
# We must transform (coder) category data, for this we will need a new library
from sklearn.preprocessing import LabelEncoder
le_X = LabelEncoder()
X[:,3] = le_X.fit_transform(X[:,3])


# Variable Dummys are category variables that does not have an order 
#Conceps: Category variable, Variable ordinales
#The hotecnoder makes the dummys variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough'                        
)

X = np.array(ct.fit_transform(X), dtype=np.float)
labelencoder_y=LabelEncoder()
#We should avoid the trick of the dummy variables
X = X[:,1:]

#video #3
# Over fitting, is when the algorithm learns certain behaviour by memory
#Divide the datatset betweent the training set and the testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)



#Multiple linear regresion

#Ajust the multiple linear regression with the training set
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)

#Prediction of the results in the testing set
y_prediction=regression.predict(X_test)


#Find an optimize set of variables among the variables selected to do the best prediction prossible
#Build the optimize model of the Multiple Linear Regression using the Backward elimination.
# In the formula of the MLR there are constants and coeficients, to decide if a variable stays in the model
# -the constant must be grather than 0 and not close to it, because if it does it means that variable does not make 
# - any difference in the model to predict the dependent variable
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)). astype(int), values =X, axis=1)


#Step 1: Choose a Significant Level that will remain in the model , in this case we will work with SL=0.05
SL = 0.05
#Step 2 Calculate the model with all possible predict variables
#Only the indepents variables  that are statistically significative to be able to predict the dependet variable
X_optimize = X[:, [0,1,2,3,4,5]].tolist()
regression_OLS= sm.OLS(endog = Y, exog = X_optimize).fit()
regression_OLS.summary()

#Step 3: Consider the predict variable with P-Value greater than the significant level
X_optimize = X[:, [0,1,3,4,5]].tolist()
regression_OLS= sm.OLS(endog = Y, exog = X_optimize).fit()
regression_OLS.summary()


X_optimize = X[:, [0,3,4,5]].tolist()
regression_OLS= sm.OLS(endog = Y, exog = X_optimize).fit()
regression_OLS.summary()


X_optimize = X[:, [0,3,5]].tolist()
regression_OLS= sm.OLS(endog = Y, exog = X_optimize).fit()
regression_OLS.summary()


X_optimize = X[:, [0,3]].tolist()
regression_OLS= sm.OLS(endog = Y, exog = X_optimize).fit()
regression_OLS.summary()








