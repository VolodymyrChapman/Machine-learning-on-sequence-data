#Consult https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

import pickle
import pandas as pd

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

#PCA 2D on whole dataset
from sklearn.decomposition import PCA
pca2d = PCA(n_components=2)
principalComponents2d = pca2d.fit_transform(X)
pca2d.explained_variance_ratio_

principalDf = pd.DataFrame(data = principalComponents2d
             , columns = ['principal component 1', 'principal component 2'], index = rownames)
finalDf = pd.concat([principalDf, Y], axis = 1)

# plotting
import matplotlib as plt
fig = plt.pyplot.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ax.set_xlim([-25, 25])
ax.set_ylim([-20, 30])
targets = [0, 1, 2, 3, 4, 5]
colors = ['y','k','b','g','m','r']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['colour'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
