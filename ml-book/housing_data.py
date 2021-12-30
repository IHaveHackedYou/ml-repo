import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np

# import data
housing = pd.read_csv("handson-ml2-master/handson-ml2-master/datasets/housing/housing.csv")
# display in pd functions all columns
pd.set_option("display.max_columns", None)
housing_optimized = housing.copy()
housing_optimized["INLAND"] = 2


data = []
for index, row in housing.iterrows():
    if row["ocean_proximity"] == "INLAND":
        data.append(1)
    else:
        data.append(0)

df = pd.DataFrame(data, columns=["INLAND"])
housing_optimized["INLAND"] = data
print(housing_optimized.head())