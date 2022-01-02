import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import MLLib as ml
from sklearn.datasets import load_iris
'''
flowers = load_iris()
data = flowers.data
df = pd.DataFrame(data)

df["target"] = flowers.target

X = df.drop("target", axis="columns")
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
previous_results = []
for i in range(100):
    model = RandomForestClassifier(n_estimators=i + 1)
    model.fit(X_train, y_train)
    current_result = model.score(X_test, y_test)
    previous_results.append(current_result)

print(previous_results)
'''

flowers = load_iris()
data = pd.DataFrame(flowers.data)
y = flowers.target
data["target"] = y
# ml.DataFrameExplorer.basic_exploration(data, plot=False)
# data_train, data_test= ml.MLPrepare.split_data(data, data["target"], n_splits=10, test_size=0.1)
X = data.drop("target", axis=1)
print(X.head())
X_scaled = pd.DataFrame(ml.MLPrepare.feature_scaling(X))
print(X_scaled.head())
