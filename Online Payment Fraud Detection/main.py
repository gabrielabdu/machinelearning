import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras
from sklearn.metrics import ConfusionMatrixDisplay

# If using Jupyter Notebook
%matplotlib inline

# 1. Load Data
try:
    data = pd.read_csv('new_data.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'new_data.csv' not found. Please check the file path.")
    # creating dummy data so the code runs for demonstration if file is missing
    data = pd.DataFrame({
        'step': np.random.randint(1, 100, 1000),
        'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT'], 1000),
        'amount': np.random.random(1000) * 1000,
        'nameOrig': ['C' + str(i) for i in range(1000)],
        'nameDest': ['M' + str(i) for i in range(1000)],
        'isFraud': np.random.choice([0, 1], 1000, p=[0.9, 0.1])
    })

# 2. Data Info & Checks
data.info()
print(data.describe())

# Check Categorical Variables
obj_cols = data.select_dtypes(include=['object']).columns
print("Categorical variables:", len(obj_cols))

# Check Integer Variables
int_cols = data.select_dtypes(include=['int64', 'int32']).columns
print("Integer variables:", len(int_cols))

# Check Float Variables
float_cols = data.select_dtypes(include=['float64', 'float32']).columns
print("Float variables:", len(float_cols))

# 3. Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 5))
sns.countplot(x='type', data=data)
plt.title('Count of Transaction Types')
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x='type', y='amount', data=data)
plt.title('Amount per Transaction Type')
plt.show()

print("Fraud Distribution:\n", data['isFraud'].value_counts())

plt.figure(figsize=(15, 6))
# 'distplot' is deprecated, using histplot with kde=True
sns.histplot(data['step'], bins=50, kde=True)
plt.title('Distribution of Steps')
plt.show()

# Correlation Heatmap
# Note: Dropping ID columns (nameOrig, nameDest) as they are high cardinality and noise
plt.figure(figsize=(12, 6))
numeric_data = data.drop(['nameOrig', 'nameDest'], axis=1).copy()
# Factorize categorical columns for correlation check
for col in numeric_data.select_dtypes(include='object').columns:
    numeric_data[col] = pd.factorize(numeric_data[col])[0]

sns.heatmap(numeric_data.corr(),
            cmap='BrBG',
            fmt='.2f',
            linewidths=2,
            annot=True)
plt.title('Correlation Matrix')
plt.show()

# 4. Feature Engineering (One-Hot Encoding)
# dtype=int ensures we get 0/1 instead of True/False
type_new = pd.get_dummies(data['type'], drop_first=True, dtype=int)
data_new = pd.concat([data, type_new], axis=1)

# Drop columns not needed for training
# We drop 'type' (since we encoded it) and the ID columns
X = data_new.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
y = data_new['isFraud']

print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)



[Image of ROC Curve explanation]


# 6. Model Training & Evaluation
# Added max_iter to LogisticRegression to prevent convergence warnings
models = [
    LogisticRegression(max_iter=1000), 
    XGBClassifier(eval_metric='logloss'), # eval_metric silences warnings
    RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
]

for model in models:
    model.fit(X_train, y_train)
    print(f'{model.__class__.__name__} : ')
    
    # Training ROC-AUC
    train_preds = model.predict_proba(X_train)[:, 1]
    print('Training ROC-AUC : ', ras(y_train, train_preds))
    
    # Validation ROC-AUC
    y_preds = model.predict_proba(X_test)[:, 1]
    print('Validation ROC-AUC : ', ras(y_test, y_preds))
    print('-' * 30)

# 7. Confusion Matrix Visualization (for XGBoost)
print("Plotting Confusion Matrix for XGBoost:")


[Image of Confusion Matrix structure]


# Using from_estimator is the modern, cleanest way
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(models[1], X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix - XGBoost')
plt.show()
