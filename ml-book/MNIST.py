import numpy as np
import pandas as pd
import MLLib as ml
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier


def import_data():
    # mnist handwritten digits dataset, content:
    # ['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url']
    mnist = fetch_openml("mnist_784", version=1)
    X, y = mnist["data"], mnist["target"]
    X["target"] = y
    X, X_bin = ml.MLPrepare.split_data(X, y, test_size=0.9)
    X["target"].to_csv("y.csv", index=False)
    X = X.drop("target", axis=1)
    X.to_csv("X.csv", index=False)


# import_data()
X = pd.DataFrame(pd.read_csv("X.csv"))
y = pd.DataFrame(pd.read_csv("y.csv"))


X = pd.read_csv("X.csv")
y = pd.read_csv("y.csv")

# conver y strings to intergers
y = y.astype(np.uint8)

# split train test split
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# if 5 is true else it is false
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

some_digit = X.iloc[0]
# Stochastic Gradient Descent
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


def cross_validation(model, X_train, y_train):
    skfolds = StratifiedKFold(n_splits=3)
    for train_index, test_index in skfolds.split(X_train, y_train):
        model = clone(model)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train[train_index]
        X_test_fold = X_train[test_index]
        y_test_fold = y_train[test_index]
        model.fit(X_train_folds, y_train_folds)
        y_pred = model.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))


cross_validation(sgd_clf, X_train, y_train_5)

