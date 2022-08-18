# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:58:32 2022

@author: Migue
"""

# Plantilla de Pre Procesado datos faltantes

# How can we import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values
#clean data, NAs
#clean data, NAs
#Library to process data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(X[:,1:4])
X[:,1:4] = imputer.transform(X[:,1:4]) 