# -*- coding: utf-8 -*-
"""
@author: Eduardo Morales Muñoz
@author: Rubén Girela Castell
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

#pipeline
from sklearn.pipeline import make_pipeline

#modelos
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

#para medir el tiempo que tarda cada modelo
from time import time

def leer(archivo):
    data = np.genfromtxt(archivo, delimiter=',')
    c = data.shape[1]-1
    Y = data[1:,c]
    X = data[1:,2:c]
    return X, Y
    

np.random.seed(1)

#%% Leo los datos


X, Y= leer("datos/OnlineNewsPopularity.csv")
for i in range(Y.size):
    if Y[i]<1400:
        Y[i]=0
    else:
        Y[i]=1

#%% División de los datos

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

#%% Correlacion entre las variables

#Nota: Si se ejecuta esta celda sola al final se muestra un grafico de más, pero cuando se ejecuta el código de corrido es 
#      necesaria la última línea de código ya que si no se combina este gráfico con el siguiente
df = pd.DataFrame(X_train)
corr = df.corr()
sns.heatmap(corr)
sns.relplot()





#%% Media y varianza

var = np.var(X_train, axis=0).mean() 
media = np.mean(X_train, axis=0).mean()

print("Sin ajustar los datos")
print('Varianza: ', var)
print('Media: ', media)
#%% Standarizacion
####
#### NORMALIZACIÓN
####

scaler = StandardScaler()

#Ajusto el normalizador a mi muestra para que calcule la media y la desviación típica
scaler.fit(X_train)

X_train = scaler.transform(X_train)

#%% Media y varianza post standar

var = np.var(X_train, axis=0).mean() 
media = np.mean(X_train, axis=0).mean()
print("\nCon los datos ajustados")
print('Varianza: ', var)
print('Media: ', media)
#%% Reducción de dimensionalidad
columnas = X_train[:, 11:17]
columnas = np.append(columnas, X_train[:, 29:36], axis=1)

X_train = np.delete(X_train, [11,12,13,14,15,16,29,30,31,32,33,34,35], axis=1)


pca = PCA(0.95, svd_solver='full')

X_train = pca.fit_transform(X_train, Y_train)

X_train = np.append(X_train, columnas, axis=1)

#%% Embedd 2D
X_embedded = TSNE(n_components=2).fit_transform(X_train)

#%% Visualizacion 2D
plt.scatter(X_embedded[:,0],X_embedded[:,1], c=Y_train)
plt.show()

#%% Optimizacion de parametros

#array de valores de parametros razonables para la obtencion de parametros
print("comprobando parametros a usar: ")
print("--------------------------------------------------------------------------------------------------")
param_grid = {'penalty': ['l1', 'l2'], 'dual': [False], 'C':[0.1, 0.5, 1,  5, 10], 'random_state': [0]}
modelo = GridSearchCV(svm.LinearSVC(), param_grid, scoring='roc_auc')
modelo.fit(X_train, Y_train)
print("--------------------------------------------------------------------------------------------------")
print('Mejores parámetros del modelo LinearSVC: ', modelo.best_params_)

#%% Linear SVC
#creamos el modelo
clf = svm.LinearSVC(C=0.5, dual=False, random_state=0)

#y entrenamos el modelo con la muestra dada
results = cross_validate(svm.LinearSVC(), X_train, Y_train, cv=5, n_jobs=-1, scoring='roc_auc')
print('AUC  en Cross-Validation con LinearSVC (parametros por defecto)', results['test_score'].mean())
results = cross_validate(clf, X_train, Y_train, cv=5, n_jobs=-1, scoring='roc_auc')
print('AUC  en Cross-Validation con LinearSVC (sin los parametros por defecto)', results['test_score'].mean())

#%% Optimizacion parametros NN
print("--------------------------------------------------------------------------------------------------1")
param_grid = {'hidden_layer_sizes': [[50, 50], [50, 60], [50, 70], [50, 80], [50, 90], [50, 100],
                                [60,50], [60,60], [60,70], [60,80], [60,90], [60,100], 
                                [70, 50], [70, 60], [70, 70], [70, 80], [70, 90], [70, 100], 
                                [80,50], [80,60], [80,70], [80,80], [80,90], [80,100], 
                                [90,50], [90,60], [90,70], [90,80], [90,90], [90,100],
                                [100,50], [100,60], [100,70], [100,80], [100,90], [100,100], ] }

modelo = GridSearchCV(MLPClassifier(), param_grid, scoring='roc_auc', n_jobs=-1)
modelo.fit(X_train, Y_train)
print("--------------------------------------------------------------------------------------------------")

print('Mejores parámetros (neuronas por capa) del Perceptron de 3 capas: ', modelo.best_params_)

#Ajustamos un poco más
print("--------------------------------------------------------------------------------------------------")
param_grid = {'hidden_layer_sizes': [[50, 50], [50,55], [55,50], [52,55], [55,52], [55,55]]}
modelo = GridSearchCV(MLPClassifier(), param_grid, scoring='roc_auc', n_jobs=-1)
modelo.fit(X_train, Y_train)
print("--------------------------------------------------------------------------------------------------")
print('Mejores parámetros (neuronas por capa) del Perceptron de 3 capas: ', modelo.best_params_)
print("--------------------------------------------------------------------------------------------------")
param_grid= {'hidden_layer_sizes': [[52, 55]], 'activation':['logistic', 'tanh', 'relu'], 'alpha':[0.0001, 0.001, 0.01], 'learning_rate_init':[0.001, 0.01, 0.1], 'learning_rate':['constant', 'invscaling', 'adaptative']}
modelo = GridSearchCV(MLPClassifier(), param_grid, scoring='roc_auc', n_jobs=-1)
modelo.fit(X_train, Y_train)
print("--------------------------------------------------------------------------------------------------")

print('Mejores parámetros del Perceptron de 3 capas: ', modelo.best_params_)


#%% Multilayer perceptron
#capas=[100, 95]  0.918707766162660
clf = MLPClassifier(hidden_layer_sizes=[52, 55], activation='logistic', alpha=0.01, learning_rate_init=0.1  )
clf.fit(X_train, Y_train)
results = cross_validate(clf, X_train, Y_train, cv=5, n_jobs=-1, scoring='roc_auc')

print('Perceptrón 3 capas en Cross-Validation AUC Score', results['test_score'].mean())

#%% RandomForest
# start_time = time()
# print("--------------------------------------------------------------------------------------------------")
# param_grid = {'n_estimators': [150, 500, 1000], 'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, None], 'min_samples_split': [2, 4, 8], 'max_features': ['sqrt', 'log2']}
# modelo = GridSearchCV(RandomForestClassifier(), param_grid, scoring='roc_auc', n_jobs=-1)
# modelo.fit(X_train, Y_train)
# print("--------------------------------------------------------------------------------------------------")
# elapsed_time = time() - start_time
# print("Calculo Elapsed time: %0.10f seconds" %elapsed_time)

# print('Mejores parámetros del Random Forest: ', modelo.best_params_)

clf = RandomForestClassifier()
results = cross_validate(clf, X_train, Y_train, cv=5, n_jobs=-1, scoring='roc_auc')
print('Random Forest (parametros por defecto) en Cross-Validation AUC Score', results['test_score'].mean())

start_time = time()
clf = RandomForestClassifier(n_estimators=800, criterion='entropy', min_samples_split=8, min_samples_leaf=2)
results = cross_validate(clf, X_train, Y_train, cv=5, n_jobs=-1, scoring='roc_auc')
elapsed_time = time() - start_time
print("Calculo Elapsed time: %0.10f seconds" %elapsed_time)
print('Random Forest (parametros modificados) en Cross-Validation AUC Score', results['test_score'].mean())