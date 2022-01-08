import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder
import MLLib as mllib

# https://www.kaggle.com/c/titanic/data?select=test.csv more data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")



# mllib.DataFrameExplorer.basic_exploration(train_data, True)
pd.set_option("display.max_columns", None)
ordinal_encoder = OrdinalEncoder()
train_data_encoded = ordinal_encoder.fit_transform(train_data)
train_data_encoded = pd.DataFrame(train_data_encoded, columns=train_data.columns)
train_data_encoded = train_data_encoded.drop("Cabin", axis=1)
# mllib.DataFrameExplorer.correlations(train_data_encoded, train_data_encoded["Survived"])
simple_imputer = SimpleImputer(strategy="median")
train_data_encoded["Age"] = train_data_encoded.fillna(inplace=True)
print(train_data_encoded)

y = train_data["Survived"]
train_data = train_data.drop("Survived", axis=1)
