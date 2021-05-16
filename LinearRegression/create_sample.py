import pandas as pd
import numpy as np

n_samples = 100
X_train = np.array([i - (n_samples // 2) for i in range(n_samples)])
y_train = np.array([3*X_train[i] for i in range(n_samples)])
train = {"X_train": X_train, "y_train": y_train}
data = pd.DataFrame(train)
data.to_csv("train_data.csv")