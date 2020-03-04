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

#PCA 3D on whole dataset - var = 0.12387244, 0.09447438, 0.08748568, respectively
from sklearn.decomposition import PCA
pca3d = PCA(n_components=3)
principalComponents3d = pca3d.fit_transform(X)
pca3d.explained_variance_ratio_

principalDf3d = pd.DataFrame(data = principalComponents3d
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'], index = rownames)
finalDf3d = pd.concat([principalDf3d, Y], axis = 1)

# plotting
from mpl_toolkits import mplot3d

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
ax.set_xlim([-25, 25])
ax.set_ylim([-30, 30])
ax.set_zlim([-20, 25])
targets = [0, 1, 2, 3, 4, 5]
colors = ['y','k','b','g','m','r']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf3d['colour'] == target
    ax.scatter(finalDf3d.loc[indicesToKeep, 'principal component 1']
               , finalDf3d.loc[indicesToKeep, 'principal component 2']
               , finalDf3d.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
ax.legend(targets)
ax.view_init(40, 100)
ax.grid()
