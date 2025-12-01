import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
# Ensure the path is correct for your environment
try:
    df = pd.read_csv('/content/Tesla.csv')
except FileNotFoundError:
    print("Error: File not found. Please check the path.")
    # creating dummy data so the code runs for demonstration if file is missing
    dates = pd.date_range(start='1/1/2010', periods=1000)
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.rand(1000) * 100,
        'High': np.random.rand(1000) * 100,
        'Low': np.random.rand(1000) * 100,
        'Close': np.random.rand(1000) * 100,
        'Adj Close': np.random.rand(1000) * 100,
        'Volume': np.random.randint(1000, 10000, 1000)
    })

print("Shape:", df.shape)
print(df.head())

# 2. Exploratory Data Analysis (EDA)
plt.figure(figsize=(15, 5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

# Check for redundant column
# We only drop Adj Close if it is identical to Close to avoid errors
if 'Adj Close' in df.columns:
    if (df['Close'] == df['Adj Close']).all():
        df = df.drop(['Adj Close'], axis=1)
        print("Dropped 'Adj Close' as it is identical to 'Close'.")

features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Distribution Plots
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.histplot(df[col], kde=True) # sb.distplot is deprecated
plt.show()

# Box Plots
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.boxplot(x=df[col])
plt.show()

# 3. Feature Engineering - Dates
# IMPROVEMENT: Use pd.to_datetime instead of string split
df['Date'] = pd.to_datetime(df['Date'])
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)

# 4. Grouped Analysis
# We group by year to see trends
data_grouped = df.drop('Date', axis=1).groupby('year').mean()
plt.subplots(figsize=(20, 10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2, 2, i + 1)
    data_grouped[col].plot.bar()
    plt.title(col)
plt.show()

# 5. Feature Engineering - Strategy
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']

# Target: 1 if tomorrow's price is higher than today's, else 0
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# IMPORTANT: The shift(-1) creates a NaN (or false comparison) at the very last row.
# We must remove the last row to ensure data integrity.
df = df.iloc[:-1]

# Check balance of target
plt.figure(figsize=(6, 6))
plt.pie(df['target'].value_counts().values, 
        labels=[0, 1], autopct='%1.1f%%')
plt.title('Target Distribution')
plt.show()

# Heatmap
plt.figure(figsize=(10, 10)) 
# Select only numeric columns for correlation to avoid errors
numeric_df = df.select_dtypes(include=[np.number])
sb.heatmap(numeric_df.corr() > 0.9, annot=True, cbar=False)
plt.show()

# 6. Preprocessing and Splitting
X = df[['open-close', 'low-high', 'is_quarter_end']]
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# CRITICAL FIX: shuffle=False
# We cannot shuffle time-series data. We must train on the past to predict the future.
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X_scaled, y, test_size=0.1, random_state=2022, shuffle=False)

print(f"Train Shape: {X_train.shape}, Valid Shape: {X_valid.shape}")

# 7. Model Training and Evaluation
models = [
    LogisticRegression(), 
    SVC(kernel='poly', probability=True), 
    XGBClassifier(eval_metric='logloss') # Added eval_metric to silence warning
]

for model in models:
    model.fit(X_train, Y_train)
    
    # Calculate scores
    train_acc = metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1])
    valid_acc = metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1])
    
    print(f'{model.__class__.__name__}:')
    print(f'Training ROC-AUC  : {train_acc:.4f}')
    print(f'Validation ROC-AUC: {valid_acc:.4f}')
    print('-' * 30)

# 8. Visualizing Results (Logistic Regression)
# Note: Using the first model in the list
print("Confusion Matrix for Logistic Regression:")
ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid)
plt.show()
