# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:57:34 2022

@author: Migue
"""

# Plantilla de Pre Procesado Datos Categoricos

# How can we import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#Codificar datos categoricos
# There are two different categories: ordinal and categorica, it is ordinal when it has an order
#Category if a Colum that does not represent a number, it represents a label for the users
# We must transform (coder) category data, for this we will need a new library
from sklearn.preprocessing import LabelEncoder
le_X = LabelEncoder()
X[:,0] = le_X.fit_transform(X[:,0])

# Variable Dummys are category variables that does not have an order 
#Conceps: Category variable, Variable ordinales
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)

X = np.array(ct.fit_transform(X), dtype=np.float)
labelencoder_y=LabelEncoder()
Y=labelencoder_y.fit_transform(Y)