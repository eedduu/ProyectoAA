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
param_grid = {'penalty': ['l1', 'l2'], 'loss':['hinge', 'squared_hinge'], 'dual': [False], 'C':[0.1, 0.5, 1,  5, 10, 100], 'random_state': [0] }
modelo = GridSearchCV(svm.LinearSVC(), param_grid, scoring='roc_auc')
modelo.fit(X_train, Y_train)
print('Mejores parámetros del modelo Linear SVC: ', modelo.best_params_)


#%% Linear SVC
#creamos el modelo
clf = svm.LinearSVC(penalty='l2', C=0.5, dual=False, random_state=0)

#y entrenamos el modelo con la muestra dada
results = cross_validate(clf, X_train, Y_train, cv=5, n_jobs=-1, scoring='roc_auc')
print('AUC  en Cross-Validation', results['test_score'].mean())


