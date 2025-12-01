import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, confusion_matrix)

# 1. Load Data
try:
    data = pd.read_csv("creditcard.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# 2. Exploratory Data Analysis
print("\n--- Data Head ---")
print(data.head())

print("\n--- Data Description ---")
print(data.describe())

# Determine fraud vs valid
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlierFraction = len(fraud) / float(len(valid))
print(f"\nOutlier Fraction: {outlierFraction}")
print(f"Fraud Cases: {len(fraud)}")
print(f"Valid Transactions: {len(valid)}")

print("\n--- Amount Details: Fraudulent Transactions ---")
print(fraud.Amount.describe())

print("\n--- Amount Details: Valid Transactions ---")
print(valid.Amount.describe())

# Correlation Matrix Heatmap
corrmat = data.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.title("Correlation Matrix")
plt.show()

# 3. Data Preprocessing
X = data.drop(['Class'], axis=1)
Y = data["Class"]
print(f"\nShape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")

xData = X.values
yData = Y.values

# 4. Train/Test Split
xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size=0.2, random_state=42
)

# 5. Model Training
# Note: n_jobs=-1 uses all processor cores to speed up training
rfc = RandomForestClassifier(n_jobs=-1, random_state=42) 
print("\nTraining Random Forest Model...")
rfc.fit(xTrain, yTrain)

# 6. Predictions
yPred = rfc.predict(xTest)

# 7. Evaluation
accuracy = accuracy_score(yTest, yPred)
precision = precision_score(yTest, yPred)
recall = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
mcc = matthews_corrcoef(yTest, yPred)

print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

