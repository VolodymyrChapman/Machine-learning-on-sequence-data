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
X_unscaled = labelled.iloc[:, 0:1467]
Y = labelled['colour']

#scaling X
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X_unscaled.values)

# Splitting into training, CV and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
X_train,X_CV,y_train, y_CV = train_test_split(X_train, y_train, test_size = 0.25)

# convert to binary categories
from keras.utils import to_categorical
y_train_binary = to_categorical(y_train)
y_test_binary = to_categorical(y_test)
y_CV_binary = to_categorical(y_CV)

# Checking dims
print(X_train.shape, y_train_binary.shape)
print(X_CV.shape, y_CV_binary.shape)
print(X_test.shape, y_test_binary.shape)

#model2 - 4 layer DNN
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3000, activation = 'relu', input_dim = 1467),
    tf.keras.layers.Dense(1000, activation = 'relu'),
    tf.keras.layers.Dense(200, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation = 'softmax')
])
model2.summary()

#compile model
from tensorflow.keras.optimizers import RMSprop
model2.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr = 0.001), metrics = ['acc'])

#fit model to training set and implement validation
history2 = model2.fit(X_train, y_train_binary, epochs = 15, validation_data= (X_CV, y_CV_binary))

#plot training and val accuracy over epochs
import matplotlib.pyplot as plt
acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()

#evaluate on test data - gives same results for accuracy, recall
test_loss = model2.evaluate(X_test, y_test_binary)




#experimenting with recall and precision
from keras import backend as K
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# compile the model
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

# fit the model
history2 = model2.fit(X_train, y_train_binary, validation_data= (X_CV, y_CV_binary), epochs=10) #, verbose=0

# evaluate the model
loss, accuracy, f1_score, precision, recall = model2.evaluate(X_test, y_test_binary, verbose=0)
