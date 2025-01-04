import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = {
    "Days": [1, 2, 3, 4, 5],
    "Candies": [2, 4, 6, 8, 10]
}


df = pd.DataFrame(data)
print(df)

plt.scatter(df["Days"], df["Candies"], color="blue")
plt.title("Candies/Days")
plt.xlabel("Days")
plt.ylabel("Candies")
#plt.show()

X = df[["Days"]]
Y = df["Candies"]
model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)
print(f"W: {model.coef_}\nb: {model.intercept_}")

plt.scatter(X, Y, color="blue")
plt.plot(X, Y_pred, color="red")
plt.title("Linear Regression")
plt.xlabel("Days")
plt.ylabel("Candies")
plt.show()

new_value = [[10]]
predicted_candy = model.predict(new_value)
print(f"Predicted for the {new_value[0][0]} day: {predicted_candy[0]}")
