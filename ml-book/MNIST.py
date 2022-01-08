import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, f1_score, \
    classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import MLLib
import MLLib as ml
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from PIL import Image


def import_data():
    # mnist handwritten digits dataset, content:
    # ['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url']
    mnist = fetch_openml("mnist_784", version=1)
    X, y = mnist["data"], mnist["target"]
    X["target"] = y
    X, X_bin = ml.MLPrepare.split_data(X, y, test_size=0.95)
    y = X["target"]
    X = X.drop("target", axis=1)
    X.to_csv("X.csv", index=False)
    y.to_csv("y.csv", index=False)


def add_samples(X, y):
    for i in range(len(X)):
        X = np.append(X, [shift_image(X[i], (0, 1))], axis=0)
        y = np.append(y, [y[i]])
        X = np.append(X, [shift_image(X[i], (1, 0))], axis=0)
        y = np.append(y, [y[i]])
        X = np.append(X, [shift_image(X[i], (0, -1))], axis=0)
        y = np.append(y, [y[i]])
        X = np.append(X, [shift_image(X[i], (-1, 0))], axis=0)
        y = np.append(y, [y[i]])
    return X, y


def show_image(X, y=None):
    if not y is None:
        print("image should be: ", y)
    digit_image = X.reshape(28, 28)
    plt.imshow(digit_image, cmap="binary")
    plt.axis("off")
    plt.show()


def cross_validation(model, X_train, y_train):
    # split X,y in train and test set
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


def shift_image(img_arr, shift):
    img_shift = np.roll(img_arr, shift)
    return img_shift


import_data()
X = pd.read_csv("X.csv")
y = pd.read_csv("y.csv")
# X, y = add_samples(X.values, y.values)
# pd.DataFrame(X).to_csv("X.csv", index=False)
# pd.DataFrame(y).to_csv("y.csv", index=False)
# X = pd.read_csv("X.csv")
# y = pd.read_csv("y.csv")
# print(X.shape, y.shape)
# convert y strings to integers
y = y.astype(np.uint8)

# split train test split
X_train, X_test, y_train, y_test = X.values[:3000], X.values[3000:], y.values[:3000], y.values[3000:]
# X_train, y_train = add_samples(X_train, y_train)

# X["target"] = y
# X_train, X_test = ml.MLPrepare.split_data(X, X["target"])
# y_train = X_train["target"].values
# y_test = X_test["target"].values
# X_train = X_train.drop("target", axis=1).values
# X_test = X_test.drop("target", axis=1).values
# if 5 is true else it is false
# y_train_5 = (y_train == 5)
# y_test_5 = (y_test == 5)

some_digit = X_train[0]
# show_image(some_digit, y.values[0])

# Stochastic Gradient Descent
# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train.values, y_train_5.values.ravel())  # converts x * 1 dataFrame to 1d array

# cross_validation(sgd_clf, X_train.values, y_train_5.values.ravel())
# MLLib.Model_Rating.cross_validation(sgd_clf, X_test, y_test_5.values.ravel())

# y_train_pred = cross_val_predict(sgd_clf, X_train.values, y_train_5.values.ravel(), cv=3)
# MLLib.Model_Rating.plot_confusion_matrix(y_train_5.values.ravel(), y_train_pred)

# precs_recall_scores = precision_score(y_train_5, y_train_pred), recall_score(y_train_5, y_train_pred)

# MLLib.Model_Rating.plot_precision_recall_vs_threshold(sgd_clf, X_train.values, y_train_5.values.ravel())
# MLLib.Model_Rating.plot_roc_curve(sgd_clf, X_train.values, y_train_5.values.ravel())

# svm_clf = SVC()
# svm_clf.fit(X_train.values, y_train.values.ravel())  # uses One versus One
# some_digit_scores = svm_clf.decision_function([some_digit]) # different scores for each class (0-9)
# ovr_clf = OneVsRestClassifier(SVC())  # uses One versus All
# ovr_clf.fit(X_train.values, y_train.values.ravel())

# print(sgd_clf.decision_function([some_digit]))
# sgd_clf.fit(X_train, y_train.values.ravel())
# # print(cross_val_score(sgd_clf, X_train, y_train.values.ravel(), cv=3, scoring="accuracy"))
# # MLLib.ModelRating.plot_confusion_matrix(sgd_clf, X_train, y_train.values.ravel())
# img = Image.open("6.png").convert("L")
# test_digit = MLLib.Image.flat_image(img)
# img_array = np.array(img)
# # print(some_digit)
# # print(test_digit.astype(float))
# print(sgd_clf.predict([test_digit]))

knn_clf = KNeighborsClassifier(weights="distance", algorithm="auto", leaf_size=30, metric="minkowski", n_neighbors=4,
                               p=1)
knn_clf.fit(X_train, y_train.ravel())
# MLLib.ModelRating.plot_confusion_matrix(knn_clf, X_train, y_train.values.ravel())
# MLLib.ModelRating.cross_validation(knn_clf, X_train, y_train.values.ravel())
# param_grid = [
#     {"n_neighbors": [4, 5, 7, 10, 12, 15, 20]},
#     {"leaf_size": [2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
#     {"p": [1, 2]}
# ]
# grid_search = GridSearchCV(knn_clf, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
# grid_search.fit(X_train, y_train.values.ravel())
# print(grid_search.best_estimator_.get_params())
y_pred = knn_clf.predict(X_test)
print(classification_report(y_test.ravel(), y_pred))
# print(shift_image(some_digit, (1, 1)).shape)
# print(show_image(some_digit, y.values[0]))
