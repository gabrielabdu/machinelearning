import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

boston = fetch_openml(name="boston", version=1, as_frame=True, parser='auto')

print(boston.data.shape)
print(boston.feature_names)

data = boston.data
data['Price'] = boston.target

print(data.head(10))
print(boston.target.shape)
print(data.head())
print(data.describe())
data.info()

x = boston.data
y = boston.target

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

print("xtrain shape : ", xtrain.shape)
print("xtest shape  : ", xtest.shape)
print("ytrain shape : ", ytrain.shape)
print("ytest shape  : ", ytest.shape)

regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

y_pred = regressor.predict(xtest)

plt.scatter(ytest, y_pred, c='green')
plt.xlabel("Price: in $1000's")
plt.ylabel("Predicted value")
plt.title("True value vs predicted value : Linear Regression")
plt.show()

mse = mean_squared_error(ytest, y_pred)
mae = mean_absolute_error(ytest, y_pred)

print("Mean Square Error : ", mse)
print("Mean Absolute Error : ", mae)
