import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

heart = pd.read_csv("heart.csv")
# display in pd functions all columns
pd.set_option("display.max_columns", None)
# heart.hist(bins=100, figsize=(20, 15))

# shuffle dataset
heart = heart.sample(frac=1).reset_index(drop=True)

corr_matrix = heart.corr()
# print(corr_matrix["HeartDisease"].sort_values(ascending=False))
# print(heart.describe())
# print(heart.head())

heart_prepared = heart.drop("HeartDisease", axis=1)
heart_labels = heart["HeartDisease"]
columns_prepared = heart_prepared.columns
# ordinal encoder to make e.g. in column "sex" from "m" and "w" 1 and 0
ordinal_encoder = OrdinalEncoder()
heart_encoded = ordinal_encoder.fit_transform(heart_prepared)
heart_encoded = pd.DataFrame(heart_encoded, columns=columns_prepared)
# heart_index = heart.reset_index()
# print(heart_encoded.head())

# unify deviation
standard_scaler = StandardScaler()
heart_prepared = standard_scaler.fit_transform(heart_encoded)
heart_prepared = pd.DataFrame(heart_prepared, columns=columns_prepared)
# print(heart_prepared.head())
heart_prepared.hist(bins=100, figsize=(20, 15))

# shuffle dataset
heart_prepared = heart_prepared.sample(frac=1).reset_index(drop=True)

# split data in train and test set
X_train, X_test, y_train, y_test = train_test_split(heart_prepared, heart_labels, test_size=0.1)

lin_reg = DecisionTreeRegressor()
lin_reg.fit(X_train, y_train)
lin_predictions = lin_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, lin_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

def prepare_predict(data, model):
    ordinal_encoder = OrdinalEncoder()
    data_encoded = ordinal_encoder.fit_transform(data)
    standard_scaler = StandardScaler()
    data_prepared = standard_scaler.fit_transform(data_encoded)
    return model.predict(data_prepared)
