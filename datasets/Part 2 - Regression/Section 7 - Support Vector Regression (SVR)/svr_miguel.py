# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 16:13:43 2022

@author: Migue
"""

#Support Vector Regression
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:34:14 2019

@author: juangabriel
"""

# Plantilla de Regresión

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('House_Rent_Dataset.csv')
X = dataset.iloc[:, 3:-1].values
Y = dataset.iloc[:, 2].values


##Categorial variables without ordering
# Variable Dummys are category variables that does not have an order 
#Conceps: Category variable, Variable ordinales
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [2])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)



# Dividir el data set en conjunto de entrenamiento y conjunto de testing
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Escalado de variables
from sklearn.preprocessing import StandardScaler
SC_X = StandardScaler()
SC_Y = StandardScaler()
X = SC_X.fit_transform(X)
Y = SC_Y.fit_transform(Y.reshape(-1,1))

# Ajustar la regresión con el dataset
from sklearn.svm import SVR
regression = SVR(kernel="rbf")
regression.fit(X, Y)

#.fit is when I want to create the model and .transform is when I want to apply the model
# Predicción de nuestros modelos
#y_pred = SC_Y.inverse_transform(regression.predict(SC_X.transform(6.5)).reshape(-1,1))
y_pred = SC_Y.inverse_transform(regression.predict(SC_X.transform(6.5)).reshape(-1, 1))

# Visualización de los resultados del Modelo Polinómico
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = "red")
plt.plot(X, regression.predict(X), color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()