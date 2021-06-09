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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/5, random_state=0)

#%% Correlacion entre las variables

#Nota: Si se ejecuta esta celda sola al final se muestra un grafico de más, pero cuando se ejecuta el código de corrido es 
#      necesaria la última línea de código ya que si no se combina este gráfico con el siguiente
df = pd.DataFrame(X_train)
corr = df.corr()
sns.heatmap(corr)
sns.relplot()



#%% Embedd 2D
#X_embedded = TSNE(n_components=2).fit_transform(X_train)

#%% Visualizacion 2D
plt.scatter(X_embedded[:,0],X_embedded[:,1], c=Y_train)
plt.legend()
plt.show()

#%% Embedd 3d
X_embedded2 = TSNE(n_components=3).fit_transform(X_train)

#%% Visualizar 3D
fig = plt.figure()
ax = Axes3D(fig)
plt.scatter(X_embedded2[:,0], X_embedded2[:,1], X_embedded2[:,2], c=Y_train)
ax.set_xlim((-20, 20))
ax.set_ylim((-20,20))
ax.set_zlim((-0.025, 0.025))
plt.legend()
plt.show()

#%% Reducción de dimensionalidad

pca = PCA(0.95, svd_solver='full')

X_train = pca.fit_transform(X_train, Y_train)

#%% Embedd 2D
X_embedded3 = TSNE(n_components=2).fit_transform(X_train)

#%% Visualizacion 2D
plt.scatter(X_embedded[:,0],X_embedded[:,1], c=Y_train)
plt.legend()
plt.show()

#%% Embedd 3d
X_embedded4 = TSNE(n_components=3).fit_transform(X_train)

#%% Visualizar 3D
fig = plt.figure()
ax = Axes3D(fig)
plt.scatter(X_embedded2[:,0], X_embedded2[:,1], X_embedded2[:,2], c=Y_train)
ax.set_xlim((-20, 20))
ax.set_ylim((-20,20))
ax.set_zlim((-0.025, 0.025))
plt.legend()
plt.show()


#%% Standarizacion
####
#### NORMALIZACIÓN
####

scaler = StandardScaler()

#Ajusto el normalizador a mi muestra para que calcule la media y la desviación típica
scaler.fit(X_train)

X_train = scaler.transform(X_train)
