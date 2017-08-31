import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from keras import backend as K


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy.io


train = scipy.io.loadmat('train_32x32')
test = scipy.io.loadmat('test_32x32')

X_train = train['X']/255.0
y_train = train['y']-1
X_test = test['X']/255.0
y_test = test['y']-1

X_train_image = np.moveaxis(X_train, 3, 0)
y_train = keras.utils.to_categorical(y_train, 10)
X_test_image = np.moveaxis(X_test, 3, 0)
y_test = keras.utils.to_categorical(y_test, 10)

batch_size = 128
num_classes = 10

input_shape = (32, 32, 3)

cnn = Sequential()
cnn.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dense(num_classes, activation='softmax'))


cnn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
cnn.fit(X_train_image, y_train,batch_size=128, epochs=30, verbose=1, validation_split=.1)

score_cnn = cnn.evaluate(X_test_image, y_test)


num_classes = 10
cnn_bn = Sequential()
cnn_bn.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
cnn_bn.add(Activation("relu"))
cnn_bn.add(BatchNormalization())
cnn_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_bn.add(Conv2D(64, (3, 3)))
cnn_bn.add(Activation("relu"))
cnn_bn.add(BatchNormalization())
cnn_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_bn.add(Conv2D(128, (3, 3)))
cnn_bn.add(Activation("relu"))
cnn_bn.add(BatchNormalization())
cnn_bn.add(MaxPooling2D(pool_size=(2, 2)))


cnn_bn.add(Flatten())
cnn_bn.add(Dense(32, activation='relu'))
cnn_bn.add(Dense(num_classes, activation='softmax'))


cnn_bn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
cnn_bn.fit(X_train_image, y_train, batch_size=128, epochs=30, verbose=1, validation_split=.1)
score_bn = cnn_bn.evaluate(X_test_image, y_test)


print("Test loss For CNN: {:.3f}".format(score_cnn[0]))
print("Test Accuracy For CNN: {:.3f}".format(score_cnn[1]))
print("Test loss For BN: {:.3f}".format(score_bn[0]))
print("Test Accuracy For BN: {:.3f}".format(score_bn[1]))
