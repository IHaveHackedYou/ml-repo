import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

import MLLib
import MLLib as ml
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
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
    X["target"].to_csv("y.csv", index=False)
    X = X.drop("target", axis=1)
    X.to_csv("X.csv", index=False)


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


# import_data()
X = pd.read_csv("X.csv")
y = pd.read_csv("y.csv")

# convert y strings to integers
y = y.astype(np.uint8)

# split train test split
X_train, X_test, y_train, y_test = X[:6000], X[6000:], y[:6000], y[6000:]
# if 5 is true else it is false
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

some_digit = X.values[0]
# show_image(some_digit, y.values[0])

# Stochastic Gradient Descent
sgd_clf = SGDClassifier(random_state=42)
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
X_train = MLLib.MLPrepare.feature_scaling(X_train)
sgd_clf.fit(X_train, y_train.values.ravel())
# print(cross_val_score(sgd_clf, X_train, y_train.values.ravel(), cv=3, scoring="accuracy"))
# MLLib.ModelRating.plot_confusion_matrix(sgd_clf, X_train, y_train.values.ravel())
img = Image.open("6.png").convert("L")
test_digit = MLLib.ImageConverter.flat_image(img)
img_array = np.array(img)
# print(some_digit)
# print(test_digit.astype(float))
print(sgd_clf.predict([test_digit]))
