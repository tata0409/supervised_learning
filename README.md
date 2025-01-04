# Linear Regression
___
## Finding Linear Regression for a dataset: X-day number, Y-number of candies

This is a project in which a linear regression for dataset is found using sklearn, pandas and matplotlib to find and display the linear regression
______
## What is a Linear Regression?
+ A **line** Y=W⋅X+b that **best fits** the data 
+ **Y** - the result(number of candies)
+ **X** - what is changing(day)
+ **W** - the slope of the line
+ **b** - the y intercept of the line
### How does it work?
+ Model finds W and b that minimize the error
+ Least squares method: minimizing the sum of squares of errors
_______
## Code
### 1. Creating the dataset

```
import numpy as np
import pandas as pd

data = {
    "Days": [1, 2, 3, 4, 5],
    "Candies": [2, 4, 6, 8, 10]
    }
df = pd.DataFrame(data)
print(df)
```

### 2. Visualizing data
```
import matplotlib.pyplot as plt

plt.scatter(df["Days"], df["Candies"], color="blue")
plt.title("Графік залежності")
plt.xlabel("Days")
plt.ylabel("Candies")
plt.show()
```
### 3. Teaching the model
```
from sklearn.linear_model import LinearRegression

X = df[["Days"]]
Y = df["Candies"]

model = LinearRegression()
model.fit(X, Y)

print("Slope(W):", model.coef_)
print("Intercept(b):", model.intercept_)
```
### 4. Finding the model and using it for predictions
```
Y_pred = model.predict(X)

# Graph with the line of best fit
plt.scatter(X, Y, color="blue")
plt.plot(X, Y_pred, color="red")
plt.title("Лінійна регресія")
plt.xlabel("Days")
plt.ylabel("Candies")
plt.show()
```
### 5. Prediction for a new value
```
new_value = [[10]]
predicted_candy = model.predict(new_value)
print(f"Predicted for the {new_value[0][0]} day: {predicted_candy[0]}")
```
---
## Results
![](/Users/tetiana/PycharmProjects/Machine Learning/supervised_learning/Figure_1.png)
___
## How to run the program

1. **Download the repository**:
   ```bash
   git clone https://github.com/tata0409/supervised_learning

2.Make sure you have Python3.8+ and all the needed libraries
___

## Contacts

If you have any questions email tatachechyna@gmail.com.

Thank you for being interested in our project!

