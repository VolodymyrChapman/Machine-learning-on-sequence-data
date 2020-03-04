#Consult https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

import pickle
import pandas as pd
import numpy as np

labelled = pickle.load(open(r'C:\Users\chapmanvl\Documents\VC 2019 projects\Influenza A segment 6 NA\even_scaled_signals_labelled_v2.txt', 'rb'))

# Splitting data into X and Y
X_unscaled = labelled.iloc[:, 0:1467]
Y = labelled['colour']
rownames = X_unscaled.index.values
X_df = pd.DataFrame(X, index = rownames)

#scaled X
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X_unscaled.values)

#PCA for 1:m components  -48 components needed for 95% var
from sklearn.decomposition import PCA
variance_ratio = []
for i in range(70):
    pca = PCA(n_components = i)
    principalComponentstrain = pca.fit(X)
    variance_ratio.append(sum(pca.explained_variance_ratio_))

import matplotlib.pyplot as plt

plt.plot(range(70),variance_ratio, 'b-')
plt.xlabel('variance')
plt.ylabel('number of components')
plt.grid()

#unscaled X
#PCA for 1:m components - 42 components needed for 95% var
from sklearn.decomposition import PCA
variance_ratio = []
for i in range(70):
    pca = PCA(n_components = i)
    principalComponentstrain = pca.fit(X_unscaled)
    variance_ratio.append(sum(pca.explained_variance_ratio_))

import matplotlib.pyplot as plt

plt.plot(range(70),variance_ratio, 'b-')
plt.xlabel('variance')
plt.ylabel('number of components')
plt.grid()