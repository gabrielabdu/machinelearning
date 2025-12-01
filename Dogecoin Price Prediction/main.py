import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Load Data
# Ensure the CSV file is in the same directory
try:
    data = pd.read_csv("DOGE-USD.csv")
except FileNotFoundError:
    print("Error: DOGE-USD.csv not found. Please upload the file.")
    # Create dummy data if file not found just so code runs for demonstration
    dates = pd.date_range(start='2021-01-01', periods=100)
    data = pd.DataFrame({
        'Date': dates,
        'Close': np.random.rand(100) * 10,
        'High': np.random.rand(100) * 12,
        'Low': np.random.rand(100) * 8,
        'Volume': np.random.randint(1000, 5000, 100)
    })

# 2. Preprocessing
# infer_datetime_format is deprecated in newer pandas versions
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Check for nulls and drop
if data.isnull().values.any():
    data = data.dropna()

# 3. Exploratory Plotting
plt.figure(figsize=(20, 7))
# Groupby is only necessary if you have duplicate dates, otherwise simple plot works
x = data['Close']
x.plot(linewidth=2.5, color='b')
plt.xlabel('Date')
plt.ylabel('Close Price') # CORRECTION: Was labeled 'Volume' but plotted 'Close'
plt.title("DOGE-USD Close Price History")
plt.show()

# 4. Feature Engineering
data["gap"] = (data["High"] - data["Low"]) * data["Volume"]
data["y"] = data["High"] / data["Volume"]
data["z"] = data["Low"] / data["Volume"]
data["a"] = data["High"] / data["Low"]
data["b"] = (data["High"] / data["Low"]) * data["Volume"]

# Check correlation (numeric_only is required for newer Pandas)
correlation = data.corr(numeric_only=True)
print(abs(correlation["Close"].sort_values(ascending=False)))

# Select features
data = data[["Close", "Volume", "gap", "a", "b"]]

# 5. Train/Test Split
# NOTE: 30 rows is very small for SARIMAX. This works for syntax checking,
# but for real trading, you need much more data.
df2 = data.tail(30)
train = df2[:11]
test = df2[-19:]

print(f"Train Shape: {train.shape}, Test Shape: {test.shape}")

# 6. Model Training
# We define the endogenous (target) and exogenous (features) variables
model = SARIMAX(
    endog=train["Close"],
    exog=train.drop("Close", axis=1),
    order=(2, 1, 1)
)
results = model.fit(disp=False) # disp=False hides convergence messages
print(results.summary())

# 7. Prediction
start = len(train)
end = len(train) + len(test) - 1

predictions = results.predict(
    start=start,
    end=end,
    exog=test.drop("Close", axis=1),
    typ='levels' # Ensures predictions are in original scale
)

# 8. Visualization of Results
plt.figure(figsize=(12, 6))
test["Close"].plot(legend=True, label="Actual Close")
predictions.plot(legend=True, label="Predictions", color='red')
plt.title("SARIMAX Prediction vs Actual")
plt.show()
