import pandas as pd
import matplotlib.pyplot as plt
heart = pd.read_csv("heart.csv")
# display in pd functions all columns
pd.set_option("display.max_columns", None)

heart.hist(bins=100, figsize=(20,15))
plt.show()
