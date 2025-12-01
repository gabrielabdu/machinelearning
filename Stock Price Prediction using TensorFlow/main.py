import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
# Ensure 'all_stocks_5yr.csv' is in your directory
try:
    data = pd.read_csv('all_stocks_5yr.csv', delimiter=',', on_bad_lines='skip')
    print("Data Shape:", data.shape)
    data['date'] = pd.to_datetime(data['date'])
except FileNotFoundError:
    print("Error: 'all_stocks_5yr.csv' not found. Please upload the file.")
    # Create dummy data if file is missing just so code is runnable for demonstration
    dates = pd.date_range(start='2013-01-01', periods=1259)
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.rand(1259) * 100,
        'close': np.random.rand(1259) * 100,
        'volume': np.random.randint(1000, 10000, 1259),
        'Name': 'AAPL'
    })

# 2. Visualize Multiple Companies (Prices)
companies = ['AAPL', 'AMD', 'FB', 'GOOGL', 'AMZN', 'NVDA', 'EBAY', 'CSCO', 'IBM']

# Only plot if these companies actually exist in the data
available_companies = [c for c in companies if c in data['Name'].unique()]

if available_companies:
    plt.figure(figsize=(15, 8))
    for index, company in enumerate(available_companies, 1):
        if index > 9: break # Limit to 9 plots
        plt.subplot(3, 3, index)
        c = data[data['Name'] == company]
        plt.plot(c['date'], c['close'], c="r", label="close")
        plt.plot(c['date'], c['open'], c="g", label="open")
        plt.title(company)
        plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Visualize Multiple Companies (Volume)
    plt.figure(figsize=(15, 8))
    for index, company in enumerate(available_companies, 1):
        if index > 9: break
        plt.subplot(3, 3, index)
        c = data[data['Name'] == company]
        plt.plot(c['date'], c['volume'], c='purple')
        plt.title(f"{company} Volume")
    plt.tight_layout()
    plt.show()

# 4. Prepare Data for LSTM (Focusing on Apple)
# We filter for AAPL (or the first available company if using dummy data)
target_stock = 'AAPL'
stock_data = data[data['Name'] == target_stock]

close_data = stock_data.filter(['close'])
dataset = close_data.values
training_len = int(np.ceil(len(dataset) * 0.95))

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training_len), :]

# Create Sliding Window (60 days)
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape for LSTM: (samples, time steps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 5. Build LSTM Model


[Image of LSTM architecture diagram]


model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

# Fixed missing parentheses
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Train
history = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

# 7. Testing Phase
test_data = scaled_data[training_len - 60:, :]
x_test = []
y_test = dataset[training_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# 8. Evaluation
mse = np.mean(((predictions - y_test) ** 2))
rmse = np.sqrt(mse)
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# 9. Plotting Results
# Use .copy() to avoid SettingWithCopyWarning
train = stock_data[:training_len].copy()
test = stock_data[training_len:].copy()
test['Predictions'] = predictions

plt.figure(figsize=(10, 8))
plt.plot(train['date'], train['close'])
plt.plot(test['date'], test[['close', 'Predictions']])
plt.title(f'{target_stock} Stock Close Price Prediction')
plt.xlabel('Date')
plt.ylabel("Close Price")
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()
