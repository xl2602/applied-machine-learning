import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor


import sklearn
if sklearn.__version__.startswith('0.17'):
    from sklearn.cross_validation import train_test_split
    from sklearn.grid_search import GridSearchCV
else:
    from sklearn.model_selection import train_test_split, GridSearchCV


def plot_history(logger):
    df = pd.DataFrame(logger.history)
    df[['acc', 'val_acc']].plot()
    plt.ylabel("accuracy")
    df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
    plt.ylabel("loss")


# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


num_classes = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Model Selection For Vanila

def make_model(optimizer="adam", hidden_size=128):
    model = Sequential([
        Dense(hidden_size, input_shape=(784,), activation='relu'),
        Dense(hidden_size, activation='relu'),
        Dense(10, activation='softmax'),
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model)
param_grid = {'hidden_size': [32, 128]}

grid_vanilla = GridSearchCV(clf, param_grid=param_grid, cv=3)
grid_vanilla.fit(X_train, y_train)


hidden_size = grid_vanilla.best_params_['hidden_size']

model = Sequential([
    Dense(hidden_size, input_shape=(784,), activation='relu'),
    Dense(hidden_size, activation='relu'),
    Dense(10, activation='softmax'),
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['accuracy'])

history_vanilla = model.fit(X_train, y_train, batch_size=200, epochs=20, verbose=1, validation_split=.1)

df = pd.DataFrame(history_vanilla.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")
plt.savefig('task2_vanilla.jpg')


# evaluate
score_vanila = model.evaluate(X_test, y_test, verbose=1)


# Model Selection For Dropout
def make_model_drop(optimizer="adam", hidden_size=128, drop=0.5):
    model_dropout = Sequential([
        Dense(hidden_size, input_shape=(784,), activation='relu'),
        Dropout(drop),
        Dense(hidden_size, activation='relu'),
        Dropout(drop),
        Dense(10, activation='softmax'),
    ])

    model_dropout.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
    return model_dropout


clf = KerasClassifier(make_model_drop)
param_grid = {'drop': [0.3, 0.5]}

grid_drop = GridSearchCV(clf, param_grid=param_grid, cv=3)
grid_drop.fit(X_train, y_train)

drop = grid_drop.best_params_['drop']

model_dropout = Sequential([
    Dense(128, input_shape=(784,), activation='relu'),
    Dropout(drop),
    Dense(128, activation='relu'),
    Dropout(drop),
    Dense(10, activation='softmax'),
])

model_dropout.compile("adam", "categorical_crossentropy", metrics=['accuracy'])

history_dropout = model_dropout.fit(X_train, y_train, batch_size=200, epochs=20, verbose=1, validation_split=.1)

df = pd.DataFrame(history_dropout.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")

plt.savefig('task2_dropout.jpg')

# evaluate
score_dropout = model_dropout.evaluate(X_test, y_test, verbose=1)

print("Test loss For Vanila: {:.3f}".format(score_vanila[0]))
print("Test Accuracy For Vanila: {:.3f}".format(score_vanila[1]))
print("Best Parameter For Vanila:", grid_vanilla.best_params_)
print("Best Train Score For Vanila:", grid_vanilla.best_score_)

print("Test loss For dropout: {:.3f}".format(score_dropout[0]))
print("Test Accuracy For dropout: {:.3f}".format(score_dropout[1]))
print("Best Parameter For Drop Out:", grid_drop.best_params_)
print("Best Train Score For Drop Out:", grid_drop.best_score_)
