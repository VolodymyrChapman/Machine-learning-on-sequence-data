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

# Splitting whole data into training, CV and test data
X_df = pd.DataFrame(X, index = rownames)
X_train, X_test, y_train, y_test = train_test_split(X_df, Y, test_size=0.2)
#X_train,X_CV,y_train, y_CV = train_test_split(X_train, y_train, test_size = 0.25)

#PCA 0.95 var on training set - 29 components
from sklearn.decomposition import PCA
pca = PCA(0.95)
principalComponentstrain = pca.fit(X_train)

#transforming test and training using 0.95 components
training = pca.transform(X_train)
testing = pca.transform(X_test)

#logistic regression on data
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(training, y_train)
logisticRegr.score(testing, y_test)
