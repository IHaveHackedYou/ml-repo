import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import zlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from pandas.plotting import scatter_matrix

# import data
housing = pd.read_csv("handson-ml2-master/handson-ml2-master/datasets/housing/housing.csv")
# display in pd functions all columns
pd.set_option("display.max_columns", None)

'''
housing.hist(bins=500, figsize=(20,15))
plt.show()
print(housing.columns)


# deprecated
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio):
    # zlib.crc32(...) converts identifier to a hash (max 2**32) and checks if this number is less than 2**32 * ratio
    return zlib.crc32(np.int64(identifier)) < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    # apply test_set_check function ids column
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# train_set, test_set = split_train_test(housing, 0.2)
housing_with_id = housing.reset_index()  # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
# print(test_set.head())
'''

'''train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# cut ["median_icome] in 5 pieces with each ranges from (e.g 0 - 1.5, 1.5 - 3.0, ...)
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# create split object 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# iterate over indices and split them regarding to ["income_cat"]
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# remove ["income_cat"] column
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
'''

'''
 housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
 housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2,
             s=housing["population"] / 100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             )
plt.legend()
plt.show()


corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]
'''


def split_data():
    # cut ["median_icome] in 5 pieces with each ranges from (e.g 0 - 1.5, 1.5 - 3.0, ...)
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    # create split object
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # iterate over indices and split them regarding to ["income_cat"]
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

    # remove ["income_cat"] column
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    return strat_train_set, strat_test_set


# split data and set test and train set
strat_train_set, strat_test_set = split_data()

# create feature set -> X
housing = strat_train_set.drop("median_house_value", axis=1)
# create label set -> y
housing_labels = strat_train_set["median_house_value"].copy()

# drop text feature "ocean_proximity", to get only numerical features
housing_num = housing.drop("ocean_proximity", axis=1)
# create object to replace missing features (e.g. on feature "oceam_proximity") with median values
imputer = SimpleImputer(strategy="median")
# calculate median values
imputer.fit(housing_num)
# execute replace
X = imputer.transform(housing_num)

# create from X (np array) DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

# detach categorical feature
housing_cat = housing[["ocean_proximity"]]
# create OrdinalEncoder to convert categorical to numerical features
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# create OneHotEncoder to create "one hot encoding" for the categorical features and saves it in a
# sparse matrix (compressed matrix) with "sparse_matrix".toarray() you can print the "normal" matrix
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


def transform_data(X, num_attribs, cat_attribs):
    # Sickit style shaped class
    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room

        def fit(self, X, y=None):
            return self  # nothing else to do

        def transform(self, X):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]

            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household,
                             bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    # When you call the pipeline’s fit() method, it calls fit_transform() sequentially on
    # all transformers, passing the output of each call as the parameter to the next call until
    # it reaches the final estimator, for which it calls the fit() method.
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),  # replace missing values with median along each column
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),  # unify deviation
    ])

    # https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),  # responsible for the num columns, num pipeline gets applied on num_attribs
        ("cat", OneHotEncoder(), cat_attribs),
        # responsible for the not num columns ("ocean_proximity"), OneHotEncoder() gets applied on cat_attribs
    ])

    X_prepared = full_pipeline.fit_transform(X)
    X_prepared_tf = pd.DataFrame(X_prepared)
    return X_prepared_tf


# columns of housing_num
num_attribs = list(housing_num)
# not numerical column
cat_attribs = ["ocean_proximity"]

housing_prepared = transform_data(housing, num_attribs, cat_attribs)

# create linearRegression model
lin_reg = LinearRegression()
# train the model
lin_reg.fit(housing_prepared, housing_labels)

# evaluate linear regression
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
# print("Prediction error in average of", tree_rmse, "$")

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
# display_scores(forest_rmse_scores)

# possibilities of hyperparameters to find out which are the best for the e.g. RandomForest
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

forest_reg = RandomForestRegressor()
# apply the testing of different hyperparameters (defined in param_grid)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

feature_importance = grid_search.best_estimator_.feature_importances_
print(feature_importance)