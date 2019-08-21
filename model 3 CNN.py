# Consult https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf

import tensorflow as tf
from tensorflow import keras
import pickle
from keras import models
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Dropout

labelled = pickle.load(open(r'C:\Users\chapmanvl\Documents\VC 2019 projects\Influenza A segment 6 NA\even_scaled_signals_labelled_v2.txt', 'rb'))

# splitting into test/training sets
from sklearn.model_selection import train_test_split

# Splitting data into X and Y
X = labelled.iloc[:, 0:1467]
Y = labelled['colour']

# Splitting into training, CV and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
X_train,X_CV,y_train, y_CV = train_test_split(X_train, y_train, test_size = 0.25)

# reshaping for conv1d to work (requires 3d array for X (n_samples, 1, n_features) 
X_t_np = X_train.to_numpy()
nrows, ncols = X_t_np.shape
X_train_reshape = X_t_np.reshape(nrows, 1, ncols)

X_cv_np = X_CV.to_numpy()
nrows, ncols = X_cv_np.shape
X_CV_reshape = X_cv_np.reshape(nrows,1, ncols )

##### x_train needs to be reshaped

# convert y to binary categories
from keras.utils import to_categorical
y_train_binary = to_categorical(y_train)
y_test_binary = to_categorical(y_test)
y_CV_binary = to_categorical(y_CV)

# Checking dims
print(X_train_reshape.shape, y_train_binary.shape)
print(X_CV_reshape.shape, y_CV_binary.shape)
###print(X_test.shape, y_test_binary.shape) - xtrain needs to be reshaped and input here

#model3 - convolution layers - dimensions not correct
model3 = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation = 'relu', input_shape = (None, 1467)), 
    tf.keras.layers.MaxPooling1D(2),
    #tf.keras.layers.Conv1D(64, 3, activation = 'relu'),
    #tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dense(72, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation = 'softmax')
])
model3.summary()

#compile model
from tensorflow.keras.optimizers import RMSprop
model3.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr = 0.001), metrics = ['acc'])

#fit model to training set and implement validation
history3 = model3.fit(X_train_reshape, y_train_binary, epochs = 8, validation_data= (X_CV_reshape, y_CV_binary))

#evaluate on test data
test_loss = model3.evaluate(X_test, y_test_binary)

#plot training and val accuracy over epochs
import matplotlib.pyplot as plt
acc = history3.history['acc']
val_acc = history3.history['val_acc']
loss = history3.history['loss']
val_loss = history3.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()