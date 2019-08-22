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
X_train_reshape = X_t_np.reshape(nrows, ncols,1)

X_cv_np = X_CV.to_numpy()
nrows, ncols = X_cv_np.shape
X_CV_reshape = X_cv_np.reshape(nrows, ncols,1 )

X_test_np = X_test.to_numpy()
nrows, ncols = X_test_np.shape
X_test_reshape = X_test_np.reshape(nrows, ncols,1 )

# convert y to binary categories
from keras.utils import to_categorical
y_train_binary = to_categorical(y_train)
y_test_binary = to_categorical(y_test)
y_CV_binary = to_categorical(y_CV)

# Checking dims
print(X_train_reshape.shape, y_train_binary.shape)
print(X_CV_reshape.shape, y_CV_binary.shape)
print(X_test_reshape.shape, y_test_binary.shape) 

#model4 - 1 convolution layer 
model4 = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation = 'relu', input_shape = (1467,1)), 
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(72, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation = 'softmax')
])
model4.summary()

#compile model
from tensorflow.keras.optimizers import RMSprop
model4.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr = 0.001), metrics = ['acc'])

#fit model to training set and implement validation
history4 = model4.fit(X_train_reshape, y_train_binary, epochs = 10, validation_data= (X_CV_reshape, y_CV_binary))

#evaluate on test data
test_loss = model4.evaluate(X_test_reshape, y_test_binary)

#plot training and val accuracy over epochs
import matplotlib.pyplot as plt
acc = history4.history['acc']
val_acc = history4.history['val_acc']
loss = history4.history['loss']
val_loss = history4.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()