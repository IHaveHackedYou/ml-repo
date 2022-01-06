import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from PIL import Image


class DataFrameExplorer:
    # need pandas DataFrame
    @staticmethod
    def basic_exploration(data, plot=True):
        pd.set_option("display.max_columns", None)
        print("data head: ")
        print(data.head(), "\n")
        print("data describe")
        print(data.describe(), "\n")
        print("data info")
        print(data.info(), "\n")
        if plot:
            data.hist(bins=50, figsize=(20, 15))
            plt.show()

    # need pandas DataFrame and String
    # prints correlations between columns
    @staticmethod
    def correlations(data, column_to_compare):
        corr_matrix = data.corr()
        return corr_matrix[column_to_compare].sort_values(ascending=False)


class MLPrepare:
    # split data in train and test set with stratifiedshufflesplit it retains same ratio of different classifications,
    # so in the test set is of every category the same ratio as in the train_set
    @staticmethod
    def split_data(X, y, n_splits=1, test_size=0.2, random_state=42):
        # n_splits is number of split iteration
        split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        strat_train_set = 0
        strat_test_set = 0
        for train_index, test_index in split.split(X, y):
            strat_train_set = X.loc[train_index]
            strat_test_set = X.loc[test_index]
        return strat_train_set, strat_test_set

    @staticmethod
    def feature_scaling(X):
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),  # replace missing values with the "median" value
            ("std_scaler", StandardScaler()),  # equal deviations and set values to a range from -2 to 2 (approx.)
        ])
        return pipeline.fit_transform(X)


class ModelRating():

    # evaluate model on different subsets
    @staticmethod
    def cross_validation(model, X_test, y_test):
        scores = cross_val_score(model, X_test, y_test, cv=3, scoring="accuracy")
        print(scores)
        return scores


    # confusion matrix shows correlations between true positive, false positive etc.
    # the column in the upper left (true positive) and lower right (true negative) should be the highest
    @staticmethod
    def plot_confusion_matrix(model, X, y, set_diagonal_zero=False):
        y_predicted = cross_val_predict(model, X, y, cv=3)
        conf_mx = confusion_matrix(y, y_predicted)
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        np.fill_diagonal(norm_conf_mx, 0)
        print_df = pd.DataFrame(norm_conf_mx)
        fig = plt.figure(figsize=(10, 7))
        sn.heatmap(print_df, annot=True)
        plt.title("Confusion Matrix")
        plt.xlabel("Real Class")
        plt.ylabel("Predicted Class")
        plt.show()

    @staticmethod
    def plot_precision_recall_vs_threshold(model, X, y):
        y_scores = cross_val_predict(model, X, y, cv=3, method="decision_function")
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

        fig = plt.figure(figsize=(5, 4))
        # Adds axes with a left, bottom, width and height that ranges from 0 to 1
        # which is the percent of the canvas you want to use
        axes_1 = fig.add_axes([0.1, 0.1, 0.9, 0.9])
        axes_1.set_xlabel("Threshold", color="red")
        axes_1.set_title("Precision vs. Recall")
        axes_1.plot(thresholds, precisions[:-1], "b--", label="Precision")
        axes_1.plot(thresholds, recalls[:-1], "g-", label="Recall")
        axes_1.legend(loc=0)
        axes_1.grid(True)
        plt.show()

    @staticmethod
    def plot_roc_curve(model, X, y, y_scores=None):
        if y_scores is None:
            y_scores = cross_val_predict(model, X, y, cv=3, method="decision_function")
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_scores)
        fig = plt.figure(figsize=(5, 4))
        axes_1 = fig.add_axes([0.1, 0.1, 0.9, 0.9])
        # plot roc curve
        axes_1.plot(false_positive_rate, true_positive_rate, linewidth=2)
        # plot diagonal
        axes_1.plot([0, 1], [0, 1], "k--")
        axes_1.set_xlabel("False Positive Rate")
        axes_1.set_ylabel("True Positive Rate")
        axes_1.grid(True)
        plt.show()


class ImageConverter():
    @staticmethod
    def flat_image(img):
        img_arr = np.array(img)
        flat_img = img_arr.ravel()
        return flat_img
