import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('RELIANCE.csv')

df.index = pd.to_datetime(df['Date'])
df = df.drop(['Date'], axis='columns')

df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low

X = df[['Open-Close', 'High-Low']]

y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

split_percentage = 0.8
split = int(split_percentage * len(df))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

cls = SVC().fit(X_train, y_train)

print("The data was split into training and testing sets using an 80/20 split.")

train_accuracy = accuracy_score(y_train, cls.predict(X_train))
test_accuracy = accuracy_score(y_test, cls.predict(X_test))

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

cls_linear = SVC(kernel='linear').fit(X_train, y_train)
y_pred_linear = cls_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f"Accuracy with Linear Kernel: {accuracy_linear}")

cls_poly = SVC(kernel='poly', degree=3).fit(X_train, y_train)
y_pred_poly = cls_poly.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print(f"Accuracy with Polynomial Kernel (degree=3): {accuracy_poly}")

cls_rbf = SVC(kernel='rbf').fit(X_train, y_train)
y_pred_rbf = cls_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"Accuracy with RBF Kernel: {accuracy_rbf}")

cls_sigmoid = SVC(kernel='sigmoid').fit(X_train, y_train)
y_pred_sigmoid = cls_sigmoid.predict(X_test)
accuracy_sigmoid = accuracy_score(y_test, y_pred_sigmoid)
print(f"Accuracy with Sigmoid Kernel: {accuracy_sigmoid}")

df['Predicted_Signal'] = cls.predict(X)

df['Return'] = df.Close.pct_change()

df['Strategy_Return'] = df.Return * df.Predicted_Signal.shift(1)

df['Cum_Ret'] = df['Return'].cumsum()

df['Cum_Strategy'] = df['Strategy_Return'].cumsum()

plt.plot(df['Cum_Ret'], color='red', label='Market Returns')
plt.plot(df['Cum_Strategy'], color='blue', label='Strategy Returns')
plt.legend()
plt.show()
