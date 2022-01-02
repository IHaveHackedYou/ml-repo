import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
