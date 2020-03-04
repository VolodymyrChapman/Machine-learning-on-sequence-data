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
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
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


# Splitting into training, CV and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
X_train,X_CV,y_train, y_CV = train_test_split(X_train, y_train, test_size = 0.25)

#PCA 2D on training set
pcatrain = PCA(n_components=2)
principalComponentstrain = pcatrain.fit_transform(X_train)
rownamestrain = X_unscaled.index.values #stopped here - need to label these somehow
principalDftrain = pd.DataFrame(data = principalComponentstrain
             , columns = ['principal component 1', 'principal component 2'], index = rownames)
finalDf = pd.concat([principalDf, Y], axis = 1)



'''
# convert to binary categories
from keras.utils import to_categorical
y_train_binary = to_categorical(y_train)
y_test_binary = to_categorical(y_test)
y_CV_binary = to_categorical(y_CV)

# Checking dims
print(X_train.shape, y_train_binary.shape)
print(X_CV.shape, y_CV_binary.shape)
print(X_test.shape, y_test_binary.shape)
'''