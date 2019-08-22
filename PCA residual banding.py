#Consult https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

import pickle
import pandas as pd
import numpy as np

labelled = pickle.load(open(r'C:\Users\chapmanvl\Documents\VC 2019 projects\Influenza A segment 6 NA\even_scaled_signals_labelled_v2.txt', 'rb'))

# splitting into test/training sets
from sklearn.model_selection import train_test_split

# Splitting data into X and Y
X_unscaled = labelled.iloc[:, 0:1467]
Y = labelled['colour']
rownames = X_unscaled.index.values

from sklearn.decomposition import PCA
# Back to original sequence to find which parts are highlighted by PCA - little detail seen
# Original Image
X1 = pd.DataFrame.to_numpy(X_unscaled) # to numpy
fig1 = plt.figure(figsize=(10, 60))
plt.imshow(X1.reshape(70,1467),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255))
plt.xlabel('1467 components', fontsize = 14)
plt.title('Original Image', fontsize = 20);

#42 components - when unscaled (sum variance = 0.9520663759975987)
pca = PCA(0.95)
fitted = pca.fit(X_unscaled)
#sum(pca.explained_variance_ratio_)
#pca.n_components_

var_95 = pca.fit_transform(X_unscaled)
approximation = pca.inverse_transform(var_95)
fig2 = plt.figure(figsize=(10, 60))
plt.imshow(approximation.reshape(70,1467),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255))
plt.xlabel('42 components', fontsize = 14)
plt.title('95% of Explained Variance', fontsize = 20)

# original minus approximation to find unimportant (residual) components
useless = pd.DataFrame.to_numpy(X_unscaled - approximation)
useless = np.where(useless > 0, useless, 0)
print(useless)
fig3 = plt.figure(figsize=(10, 60))
plt.imshow(useless.reshape(70,1467),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255))
plt.xlabel('1467 - 42 components', fontsize = 14)
plt.title('useless(residual) bands', fontsize = 20)