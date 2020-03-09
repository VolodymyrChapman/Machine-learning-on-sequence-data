
# Script to identify which even-scaled signals are responsible for 95% variance
# in the Influenza Neuraminidase dataset (order of sequences and labelling available
# in Key to labelling of columns in output dataframe.txt file)

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing even-scaled data - data stored in 'pickle' format 
# within even_scaled_signals_labelled_v2.txt file hosted in 
#machine-learning-on-sequence-data repository of my GitHub
labelled = pickle.load(open(r'E:\Programming\GitHub\machine-learning-on-sequence-data\even_scaled_signals_labelled_v2.txt', 'rb'))

# splitting into test/training sets
from sklearn.model_selection import train_test_split

# Splitting data into X and Y
X_unscaled = labelled.iloc[:, 0:1467]
Y = labelled['colour']
rownames = X_unscaled.index.values



#PCA
from sklearn.decomposition import PCA

# Original even-scaled data
X1 = pd.DataFrame.to_numpy(X_unscaled) # to numpy

#plotting
fig1 = plt.figure(figsize=(10, 60))
plt.imshow(X1.reshape(70,1467),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255))
plt.xlabel('1467 components', fontsize = 14)
plt.title('Original Image', fontsize = 20);


#Retrieving components responsible for 95% variance (sum variance = 0.9520663759975987)
pca = PCA(0.95)
fitted = pca.fit(X_unscaled)
#sum(pca.explained_variance_ratio_)
#pca.n_components_
var_95 = pca.fit_transform(X_unscaled)
approximation = pca.inverse_transform(var_95)

#plotting
fig2 = plt.figure(figsize=(10, 60))
plt.imshow(approximation.reshape(70,1467),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255))
plt.xlabel('42 components', fontsize = 14)
plt.title('95% of Explained Variance', fontsize = 20)


# (original data * 95% variance data) to identify signals accounting for 95% var 
useful = pd.DataFrame.to_numpy(X_unscaled * approximation)
print(useful)
fig3 = plt.figure(figsize=(10, 60))
plt.imshow(useful.reshape(70,1467),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255))
plt.xlabel('1467 - 42 components', fontsize = 14)
plt.title('Signals responsible for 95% variance', fontsize = 20)