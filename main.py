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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

#pipeline
from sklearn.pipeline import make_pipeline

#modelos
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

#Dummy model para comparar
from sklearn.dummy import DummyClassifier

#para medir el tiempo que tarda cada modelo
from time import time

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

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
    if Y[i]<=1400:
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
# X_embedded = TSNE(n_components=2).fit_transform(X_train)

# #%% Visualizacion 2D
# plt.scatter(X_embedded[:,0],X_embedded[:,1], c=Y_train)
# plt.show()

#%% Optimizacion de parametros

#array de valores de parametros razonables para la obtencion de parametros
# print("comprobando parametros a usar: ")
# print("--------------------------------------------------------------------------------------------------")
# param_grid = {'penalty': ['l2'], 'loss':['hinge', 'squared_hinge'], 'dual': [False], 'random_state': [0]}
# modelo = GridSearchCV(svm.LinearSVC(), param_grid, scoring='roc_auc')
# modelo.fit(X_train, Y_train)
# print("--------------------------------------------------------------------------------------------------")
# print('Mejores parámetros del modelo LinearSVC: ', modelo.best_params_)

#%% Linear SVC

#lista de resultados para comparar
list_data = []
nombres = ["LinearSVC", "MLP", "RandomForest"]

# #creamos el modelo
# clf = svm.LinearSVC( dual=False, random_state=0)

# #y entrenamos el modelo con la muestra dada
# results = cross_validate(clf, X_train, Y_train, cv=5, n_jobs=-1, scoring='roc_auc')
# print('AUC  en Cross-Validation con LinearSVC (sin regularizacion)', results['test_score'].mean())


clf = svm.LinearSVC(C=0.00000000000000001, dual=False, random_state=0)

results = cross_validate(clf, X_train, Y_train, cv=5, n_jobs=-1, scoring='roc_auc')
print('AUC  en Cross-Validation con LinearSVC (con regularizacion)', results['test_score'].mean())
list_data.append(results['test_score'].mean());

#%% Optimizacion parametros NN
# print("--------------------------------------------------------------------------------------------------1")
# param_grid = {'hidden_layer_sizes': [[50, 50], [50, 60], [50, 70], [50, 80], [50, 90], [50, 100],
#                                 [60,50], [60,60], [60,70], [60,80], [60,90], [60,100], 
#                                 [70, 50], [70, 60], [70, 70], [70, 80], [70, 90], [70, 100], 
#                                 [80,50], [80,60], [80,70], [80,80], [80,90], [80,100], 
#                                 [90,50], [90,60], [90,70], [90,80], [90,90], [90,100],
#                                 [100,50], [100,60], [100,70], [100,80], [100,90], [100,100], ] }

# modelo = GridSearchCV(MLPClassifier(), param_grid, scoring='roc_auc', n_jobs=-1)
# modelo.fit(X_train, Y_train)
# print("--------------------------------------------------------------------------------------------------")

# print('Mejores parámetros (neuronas por capa) del Perceptron de 3 capas: ', modelo.best_params_)

# #Ajustamos un poco más
# print("--------------------------------------------------------------------------------------------------")
# param_grid = {'hidden_layer_sizes': [[50, 50], [50,55], [55,50], [52,55], [55,52], [55,55]]}
# modelo = GridSearchCV(MLPClassifier(), param_grid, scoring='roc_auc', n_jobs=-1)
# modelo.fit(X_train, Y_train)
# print("--------------------------------------------------------------------------------------------------")
# print('Mejores parámetros (neuronas por capa) del Perceptron de 3 capas: ', modelo.best_params_)
# print("--------------------------------------------------------------------------------------------------")
# param_grid= {'solver':['lbfgs', 'adam', 'sgd'], 'hidden_layer_sizes': [[52, 55]], 'activation':['logistic', 'tanh', 'relu'], 'learning_rate_init':[0.001, 0.01, 0.0001], 'learning_rate':['constant', 'invscaling', 'adaptative']}
# modelo = GridSearchCV(MLPClassifier(), param_grid, scoring='roc_auc', n_jobs=-1)
# modelo.fit(X_train, Y_train)
# print("--------------------------------------------------------------------------------------------------")

# print('Mejores parámetros del Perceptron de 3 capas: ', modelo.best_params_)


# #%% Multilayer perceptron CV
# clf = MLPClassifier(hidden_layer_sizes=[52, 55], activation='logistic', learning_rate='constant', learning_rate_init=0.01, solver='sgd'  )
# clf.fit(X_train, Y_train)
# results = cross_validate(clf, X_train, Y_train, cv=5, n_jobs=-1, scoring='roc_auc')

# print('Perceptrón 3 capas sin regularizacion en Cross-Validation AUC Score', results['test_score'].mean())



clf = MLPClassifier(hidden_layer_sizes=[52, 55], activation='logistic', learning_rate='constant', learning_rate_init=0.01, solver='sgd' , alpha=5)
clf.fit(X_train, Y_train)
results = cross_validate(clf, X_train, Y_train, cv=5, n_jobs=-1, scoring='roc_auc')

print('Perceptrón 3 capas con regularizacion en Cross-Validation AUC Score', results['test_score'].mean())
list_data.append(results['test_score'].mean());

#%% RandomForest parametros
start_time = time()
# print("--------------------------------------------------------------------------------------------------")
# param_grid = {'n_estimators': [100,200,300,400, 500], 'criterion': ['gini', 'entropy'], 'n_jobs':[-1], 'random_state':[0], 'bootstrap':[True, False], 'oob_score':[True, False]}
# modelo = GridSearchCV(RandomForestClassifier(), param_grid, scoring='roc_auc', n_jobs=-1)
# modelo.fit(X_train, Y_train)
# print("--------------------------------------------------------------------------------------------------")
# elapsed_time = time() - start_time
# print("Calculo Elapsed time: %0.10f seconds" %elapsed_time)

# print('Mejores parámetros del Random Forest: ', modelo.best_params_)

# #%% RandomForest CV


# start_time = time()
# clf = RandomForestClassifier(n_estimators=500, criterion='entropy', oob_score=True, n_jobs=-1, random_state=0)
# results = cross_validate(clf, X_train, Y_train, cv=5, n_jobs=-1, scoring='roc_auc')
# elapsed_time = time() - start_time
# print("Calculo Elapsed time: %0.10f seconds" %elapsed_time)
# print('Random Forest (sin regularizacion) en Cross-Validation AUC Score', results['test_score'].mean())


start_time = time()
clf = RandomForestClassifier(n_estimators=500, criterion='entropy', oob_score=True, max_leaf_nodes=4, n_jobs=-1, random_state=0)
results = cross_validate(clf, X_train, Y_train, cv=5, n_jobs=-1, scoring='roc_auc')
elapsed_time = time() - start_time
print("Calculo Elapsed time: %0.10f seconds" %elapsed_time)
print('Random Forest (con regularizacion) en Cross-Validation AUC Score', results['test_score'].mean())
list_data.append(results['test_score'].mean());

#%% Grafica de barras de Cross-Validation
fig, ax = plt.subplots()
ax.bar(nombres,list_data)
fig.suptitle('Comparación Cross-Validation')
fig.show()


#%% Preparo datos test


scaler.transform(X_test)

columnas = X_test[:, 11:17]
columnas = np.append(columnas, X_test[:, 29:36], axis=1)

X_test = np.delete(X_test, [11,12,13,14,15,16,29,30,31,32,33,34,35], axis=1)

#pca = PCA(28)

X_test = pca.transform(X_test)

X_test = np.append(X_test, columnas, axis=1)


#%% Calculo Ein y Eout Random Forest

list_ein=[]
list_etest=[]

clf = RandomForestClassifier(n_estimators=500, criterion='entropy', oob_score=True, max_leaf_nodes=4, n_jobs=-1, random_state=0)
clf.fit(X_train, Y_train)
print('AUC score RF train', roc_auc_score(Y_train, clf.predict(X_train)))
print('Accuracy score RF  train', accuracy_score(Y_train, clf.predict(X_train)))

print('AUC score RF test', roc_auc_score(Y_test, clf.predict(X_test)))
print('Accuracy score RF test', accuracy_score(Y_test, clf.predict(X_test)))

clf = RandomForestClassifier(n_jobs=-1, random_state=0)
clf.fit(X_train, Y_train)
print('Defecto AUC score RF train', roc_auc_score(Y_train, clf.predict(X_train)))
print('Defecto Accuracy score RF  train', accuracy_score(Y_train, clf.predict(X_train)))

print('Defecto AUC score RF test', roc_auc_score(Y_test, clf.predict(X_test)))
print('Defecto Accuracy score RF test', accuracy_score(Y_test, clf.predict(X_test)))

list_ein.append(roc_auc_score(Y_train, clf.predict(X_train)))
list_etest.append(roc_auc_score(Y_test, clf.predict(X_test)))

clf = RandomForestClassifier(n_estimators=500, criterion='entropy', oob_score=True, n_jobs=-1, random_state=0)
clf.fit(X_train, Y_train)
print('SR AUC score RF train', roc_auc_score(Y_train, clf.predict(X_train)))
print('SR Accuracy score RF  train', accuracy_score(Y_train, clf.predict(X_train)))

print('SR AUC score RF test', roc_auc_score(Y_test, clf.predict(X_test)))
print('SR Accuracy score RF test', accuracy_score(Y_test, clf.predict(X_test)))

list_ein.append(roc_auc_score(Y_train, clf.predict(X_train)))
list_etest.append(roc_auc_score(Y_test, clf.predict(X_test)))



#%% DUmmy test

clf = DummyClassifier()
clf.fit(X_train, Y_train)
print('AUC score Dummy train', roc_auc_score(Y_train, clf.predict(X_train)))
print('Accuracy score Dummy  train', accuracy_score(Y_train, clf.predict(X_train)))

print('AUC score poly Dummy test', roc_auc_score(Y_test, clf.predict(X_test)))
print('Accuracy score Dummy test', accuracy_score(Y_test, clf.predict(X_test)))

list_ein.append(roc_auc_score(Y_train, clf.predict(X_train)))
list_etest.append(roc_auc_score(Y_test, clf.predict(X_test)))


#%% Calculo Ein y Eout MLP
clf = MLPClassifier(hidden_layer_sizes=[52, 55], activation='logistic', learning_rate='constant', learning_rate_init=0.01, solver='sgd' )
clf.fit(X_train, Y_train)
print('AUC score RF train', roc_auc_score(Y_train, clf.predict(X_train)))
print('Accuracy score RF  train', accuracy_score(Y_train, clf.predict(X_train)))

print('AUC score RF test', roc_auc_score(Y_test, clf.predict(X_test)))
print('Accuracy score RF test', accuracy_score(Y_test, clf.predict(X_test)))

clf = MLPClassifier(hidden_layer_sizes=[52, 55], activation='logistic', learning_rate='constant', learning_rate_init=0.01, solver='sgd' , alpha=15)
clf.fit(X_train, Y_train)
print('AUC score RF train', roc_auc_score(Y_train, clf.predict(X_train)))
print('Accuracy score RF  train', accuracy_score(Y_train, clf.predict(X_train)))

print('AUC score RF test', roc_auc_score(Y_test, clf.predict(X_test)))
print('Accuracy score RF test', accuracy_score(Y_test, clf.predict(X_test)))

#%% Grafica Ein y Etest
nombres = ['Dummy', 'RandomForest Default','RandomForest Sin Regularización']
x = np.arange(len(nombres))
width = 0.35

fig2, ax2 = plt.subplots()

rect1 = ax.bar(x-width/2, list_ein, width, label='Ein')
rect2 = ax.bar(x+width/2, list_etest, width, label='Etest')

ax2.set_xticks(x)
ax2.set_xticklabels(nombres)
ax2.legend()
ax2.bar_label(rect1, padding=3)
ax2.bar_label(rect2, padding=3)
fig2.suptitle('Comparación Ein y Etest')
fig2.tight_layout()
fig2.show()
