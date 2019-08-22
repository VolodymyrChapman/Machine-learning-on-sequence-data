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

#scaling X
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X_unscaled.values)

# Splitting whole data into training and test data
X_df = pd.DataFrame(X, index = rownames)
X_train, X_test, y_train, y_test = train_test_split(X_df, Y, test_size=0.2)


#PCA on training set
from sklearn.decomposition import PCA
rownames_train = X_train.index.values
pca_train = PCA(n_components = 2)
principalComponents_train = pca_train.fit_transform(X_train)

training = pd.DataFrame(data = principalComponents_train
             , columns = ['principal component 1', 'principal component 2'], index = rownames_train)
traindf = pd.concat([training, y_train], axis = 1)

#PCA on test set
rownames_test = X_test.index.values
pca_test = PCA(n_components = 2)
principalComponents_test = pca_test.fit_transform(X_test)

test = pd.DataFrame(data = principalComponents_test
             , columns = ['principal component 1', 'principal component 2'], index = rownames_test)
testdf = pd.concat([test, y_test], axis = 1)

# plotting training set
import matplotlib.pyplot as plt
fig_training = plt.figure(figsize = (8,8))
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.title('PCA training set', fontsize = 20)
plt.xlim([-25, 30])
plt.ylim([-25, 30])
targets = [0, 1, 2, 3, 4, 5]
colors = ['y','k','b','g','m','r']
for target, color in zip(targets,colors):
    indicesToKeep = traindf['colour'] == target
    plt.scatter(traindf.loc[indicesToKeep, 'principal component 1']
               , traindf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
plt.legend(targets)
plt.grid()

# plotting test set
import matplotlib.pyplot as plt
fig_test = plt.figure(figsize = (8,8))
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.title('PCA test set', fontsize = 20)
plt.xlim([-25, 30])
plt.ylim([-25, 30])
targets = [0, 1, 2, 3, 4, 5]
colors = ['y','k','b','g','m','r']
for target, color in zip(targets,colors):
    indicesToKeep = testdf['colour'] == target
    plt.scatter(testdf.loc[indicesToKeep, 'principal component 1']
               , testdf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
plt.legend(targets)
plt.grid()