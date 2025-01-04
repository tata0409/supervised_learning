import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


np.random.seed(24)

hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
energy_usage = 10 * hours + 15 + np.random.normal(0, 5, len(hours))
X = hours.reshape(-1, 1)
Y = energy_usage
model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)
print(f"W: {model.coef_[0]}\nb: {model.intercept_}")

plt.scatter(X, Y, color="blue")
plt.plot(X, Y_pred, color="red")
plt.title("Linear Regression")
plt.xlabel("Hours")
plt.ylabel("Energy usage")
plt.show()

