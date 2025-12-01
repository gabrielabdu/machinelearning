import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.metrics import RootMeanSquaredError

try:
    microsoft = pd.read_csv('MicrosoftStock.csv')
except FileNotFoundError:
    print("Error: CSV file not found. Please upload 'MicrosoftStock.csv'.")
    dates = pd.date_range(start='1/1/2010', periods=3000)
    microsoft = pd.DataFrame({
        'date': dates,
        'open': np.random.rand(3000) * 100,
        'close': np.cumsum(np.random.randn(3000)) + 100,
        'volume': np.random.randint(100, 1000, size=3000)
    })

print(microsoft.head())
print(microsoft.shape)

microsoft['date'] = pd.to_datetime(microsoft['date'])

plt.figure(figsize=(12, 6))
plt.plot(microsoft['date'], microsoft['open'], color="blue", label="open")
plt.plot(microsoft['date'], microsoft['close'], color="green", label="close")
plt.title("Microsoft Open-Close Stock")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(microsoft.select_dtypes(include=np.number).corr(), annot=True, cbar=False)
plt.show()

msft_close = microsoft.filter(['close'])
dataset = msft_close.values

training_data_len = int(np.ceil(len(dataset) * .95))

scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training_data_len), :]

x_train = []
y_train = []
window_size = 60

for i in range(window_size, len(train_data)):
    x_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=64))
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(1))

print(model.summary())

model.compile(optimizer='adam', loss='mae', metrics=[RootMeanSquaredError()])

history = model.fit(x_train, y_train, epochs=20, batch_size=60)

test_data = scaled_data[training_data_len - window_size:, :]

x_test = []
y_test = dataset[training_data_len:, :]

for i in range(window_size, len(test_data)):
    x_test.append(test_data[i-window_size:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

train = microsoft[:training_data_len].copy()
valid = microsoft[training_data_len:].copy()
valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('Model Evaluation: Microsoft Stock Price')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['date'], train['close'])
plt.plot(valid['date'], valid[['close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
