# Task 2.4

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import load_iris
import numpy as np


def knn_iris(neighbors):
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    cross_val_scores = []

    knn = KNeighborsClassifier(n_neighbors=neighbors)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cross_val_scores.append(np.mean(scores))

    return np.max(cross_val_scores)


def test_knn_iris():
    neighbors = range(1, 50, 1)
    for i in neighbors:
        assert knn_iris(i) >= .7



