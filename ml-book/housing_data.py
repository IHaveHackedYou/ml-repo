import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

housing = pd.read_csv("handson-ml2-master/handson-ml2-master/datasets/housing/housing.csv")
# display in pd functions all columns
pd.set_option("display.max_columns", None)
housing_prepared = housing.copy()
housing_prepared["INLAND"] = 0
housing_prepared = housing_prepared.drop("ocean_proximity", axis=1)
housing_prepared = housing_prepared.drop("median_house_value", axis=1)
housing_labels = housing["median_house_value"]
data = []
for index, row in housing.iterrows():
    if row["ocean_proximity"] == "INLAND":
        data.append(1)
    else:
        data.append(0)

df = pd.DataFrame(data, columns=["INLAND"])
housing_prepared["INLAND"] = data

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),  # replace missing values with median along each column
        ('std_scaler', StandardScaler()),  # unify deviation
    ])

housing_prepared = num_pipeline.fit_transform(housing_prepared)

X_train, X_test, y_train, y_test = train_test_split(housing_prepared, housing_labels, test_size=0.2)

'''
# create linearRegression model
lin_reg = LinearRegression()
# train the model
lin_reg.fit(X_train, y_train)

housing_predictions = lin_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
'''
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)

housing_predictions = tree_reg.predict(X_test)
tree_mse = mean_squared_error(y_test, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Prediction error in average of", tree_rmse, "$")

