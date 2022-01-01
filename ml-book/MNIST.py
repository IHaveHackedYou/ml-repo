import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# mnist handwritten digits dataset, content:
# ['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url']
mnist = fetch_openml("mnist_784", version=1)

X, y = mnist["data"], mnist["target"]

print(type(X))
# some_digit = X[0]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap="binary")
# plt.axis("off")
# plt.show()
