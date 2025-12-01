import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
# Ensure 'bitcoin.csv' is in the same directory
try:
    df = pd.read_csv('bitcoin.csv')
except FileNotFoundError:
    print("Error: 'bitcoin.csv' not found. Please upload the file.")
    # creating dummy data just for the code to run if you copy-paste it without a file
    dates = pd.date_range(start='2020-01-01', periods=1000)
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.rand(1000) * 10000,
        'High': np.random.rand(1000) * 10000,
        'Low': np.random.rand(1000) * 10000,
        'Close': np.random.rand(1000) * 10000,
        'Adj Close': np.random.rand(1000) * 10000
    })

# Basic Inspection
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())

# 2. EDA: Close Price
plt.figure(figsize=(15, 5))
plt.plot(df['Close'])
plt.title('Bitcoin Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

# Drop Adj Close if it is identical to Close
if 'Adj Close' in df.columns:
    if np.array_equal(df['Close'].values, df['Adj Close'].values):
        df = df.drop(['Adj Close'], axis=1)

# Check for nulls
print(f"Null values: \n{df.isnull().sum()}")

# Feature Distributions
features = ['Open', 'High', 'Low', 'Close']

plt.figure(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    sn.histplot(df[col], kde=True) # Updated from distplot
plt.show()

plt.figure(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    sn.boxplot(x=df[col]) # Updated orientation syntax
plt.show()

# 3. Feature Engineering
# Convert Date first for cleaner extraction
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

# Grouped Analysis
# numeric_only=True is required for pandas 2.0+
data_grouped = df.groupby('year', numeric_only=True).mean()
plt.figure(figsize=(20, 10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2, 2, i + 1)
    data_grouped[col].plot.bar()
    plt.title(col)
plt.show()

df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)

# Create input features
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']

# Create Target
# 1 if next day price is higher than today, else 0
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# IMPORTANT: The shift creates a NaN in the last row. We must remove it.
df.dropna(inplace=True)

# Check Class Balance
plt.pie(df['target'].value_counts().values, 
        labels=[0, 1], autopct='%1.1f%%')
plt.title('Target Class Balance')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 10))
# numeric_only=True ensures we don't try to correlate the 'Date' object
sn.heatmap(df.corr(numeric_only=True) > 0.9, annot=True, cbar=False)
plt.show()

# 4. Preprocessing & Splitting
feature_cols = df[['open-close', 'low-high', 'is_quarter_end']]
target_col = df['target']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(feature_cols)

# CRITICAL FIX: Time Series Split
# We cannot use random_state shuffle for Time Series. 
# We must split chronologically (Train on past, Test on future)
train_size = int(len(features_scaled) * 0.7) # 70% split

X_train = features_scaled[:train_size]
Y_train = target_col[:train_size]
X_valid = features_scaled[train_size:]
Y_valid = target_col[train_size:]

print(f"Train Size: {X_train.shape}, Valid Size: {X_valid.shape}")

# 5. Model Training & Evaluation
models = [
    LogisticRegression(), 
    SVC(kernel='poly', probability=True), 
    XGBClassifier()
]

for i in range(3):
    models[i].fit(X_train, Y_train)
    
    print(f'{models[i]} : ')
    
    # Calculate ROC AUC
    train_proba = models[i].predict_proba(X_train)[:,1]
    valid_proba = models[i].predict_proba(X_valid)[:,1]
    
    print('Training Accuracy (ROC AUC): ', metrics.roc_auc_score(Y_train, train_proba))
    print('Validation Accuracy (ROC AUC): ', metrics.roc_auc_score(Y_valid, valid_proba))
    print()

# 6. Visualization of Results
# Show confusion matrix for the first model (Logistic Regression)
ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid, cmap='Blues')
plt.title('Confusion Matrix: Logistic Regression')
plt.show()
