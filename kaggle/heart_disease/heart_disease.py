import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

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

# ordinal encoder to make e.g. in column "sex" from "m" and "w" 1 and 0
ordinal_encoder = OrdinalEncoder()
heart_encoded = ordinal_encoder.fit_transform(heart)
heart_encoded = pd.DataFrame(heart_encoded, columns=heart.columns)
# heart_index = heart.reset_index()
# print(heart_encoded.head())

# unify deviation
standard_scaler = StandardScaler()
heart_prepared = standard_scaler.fit_transform(heart_encoded)
heart_prepared = pd.DataFrame(heart_prepared, columns=heart.columns)
print(heart_prepared.head())
heart_prepared.hist(bins=100, figsize=(20, 15))
# plt.show()

housing_labels = heart.drop("HeartDisease", axis=1)

X_train, X_test, y_train, y_test = train_test_split(heart_prepared, housing_labels, test_size=0.2)

