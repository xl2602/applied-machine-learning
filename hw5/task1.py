import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

import sklearn

if sklearn.__version__.startswith('0.17'):
    from sklearn.cross_validation import train_test_split
    from sklearn.grid_search import GridSearchCV
else:
    from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import datasets


def make_model(optimizer="adam", hidden_size=32):
    model = Sequential([
        Dense(hidden_size, input_shape=(4,), activation='relu'),
        Dense(hidden_size, activation='relu'),
        Dense(3, activation='softmax'),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model


# load data
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    random_state=6, stratify=iris.target)

# one hot encoder
num_classes = 3
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Model Select
clf = KerasClassifier(make_model)
param_grid = {'epochs': [5, 10,20],
              'hidden_size': [32, 64, 128]}

grid = GridSearchCV(clf, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)



# evaluate
hidden_size = grid.best_params_['hidden_size']
epochs = grid.best_params_['epochs']
model = Sequential([
    Dense(hidden_size, input_shape=(4,), activation='relu'),
    Dense(hidden_size, activation='relu'),
    Dense(3, activation='softmax'),
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=1,
          epochs=epochs, verbose=1, validation_split=.1)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))
print("Best Parameter:", grid.best_params_)
print("Best Train Score:", grid.best_score_)