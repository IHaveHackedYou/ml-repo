import pandas as pd
import matplotlib.pyplot as plt

heart = pd.read_csv("heart.csv")
# display in pd functions all columns
pd.set_option("display.max_columns", None)
# heart.hist(bins=100, figsize=(20, 15))

# shuffle dataset
heart = heart.sample(frac=1).reset_index(drop=True)

corr_matrix = heart.corr()
# printcorr_matrix["HeartDisease"].sort_values(ascending=False)