import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error as mae
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# 1. Load Data
try:
    df = pd.read_csv('Zillow.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'Zillow.csv' not found. Please ensure the file is in the directory.")
    # Create dummy data for demonstration purposes so the code doesn't crash
    df = pd.DataFrame({
        'parcelid': range(1000),
        'bathroomcnt': np.random.randint(1, 5, 1000),
        'bedroomcnt': np.random.randint(1, 6, 1000),
        'calculatedbathnbr': np.random.randint(1, 5, 1000),
        'fullbathcnt': np.random.randint(1, 5, 1000),
        'fips': np.random.choice([6037, 6059, 6111], 1000),
        'rawcensustractandblock': np.random.rand(1000),
        'taxvaluedollarcnt': np.random.randint(100000, 1000000, 1000),
        'finishedsquarefeet12': np.random.randint(500, 5000, 1000),
        'landtaxvaluedollarcnt': np.random.randint(50000, 500000, 1000),
        'region': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.normal(0, 0.5, 1000) # Logerror
    })
    # Add some nulls to simulate real data
    df.loc[0:10, 'bathroomcnt'] = np.nan

print(f"Initial Shape: {df.shape}")

# 2. Data Cleaning
to_remove = []
for col in df.columns:
    # Remove columns with only one unique value
    if df[col].nunique() == 1:
        to_remove.append(col)
    # Remove columns with > 60% null values
    elif (df[col].isnull()).mean() > 0.60:
        to_remove.append(col)

print(f"Columns to remove: {len(to_remove)}")
df.drop(to_remove, axis=1, inplace=True)

# 3. Imputation (Filling Missing Values)
# Note: In a strict ML pipeline, fit imputer on train, apply to test to avoid leakage.
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    elif pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mean())

# Verify nulls are gone
print(f"Total Nulls: {df.isnull().sum().sum()}")

# 4. Feature Separation (Identification)
ints = []
objects = []
floats = []

for col in df.columns:
    if df[col].dtype == float:
        floats.append(col)
    elif df[col].dtype == int:
        ints.append(col)
    else:
        objects.append(col)

print(f"Ints: {len(ints)}, Floats: {len(floats)}, Objects: {len(objects)}")

# 5. Exploratory Data Analysis
plt.figure(figsize=(8, 5))
# sb.distplot is deprecated, use histplot with kde=True
sb.histplot(df['target'], kde=True)
plt.title("Distribution of Target")
plt.show()

plt.figure(figsize=(8, 5))
sb.boxplot(x=df['target'])
plt.title("Boxplot of Target")
plt.show()

# 6. Outlier Removal
print('Shape before outlier removal:', df.shape)
df = df[(df['target'] > -1) & (df['target'] < 1)]
print('Shape after outlier removal:', df.shape)

# 7. Encoding Categorical Variables
for col in objects:
    if col in df.columns: # Check existence
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# 8. Correlation Analysis
plt.figure(figsize=(10, 10))
# 'numeric_only=True' is needed for newer Pandas versions
sb.heatmap(df.corr(numeric_only=True) > 0.8, annot=True, cbar=False)
plt.title("High Correlation Heatmap")
plt.show()

# Drop correlated features manually
features_to_drop = [
    'calculatedbathnbr', 'fullbathcnt', 'fips', 
    'rawcensustractandblock', 'taxvaluedollarcnt', 
    'finishedsquarefeet12', 'landtaxvaluedollarcnt'
]
# Only drop if they actually exist in the dataframe
existing_to_drop = [col for col in features_to_drop if col in df.columns]
df.drop(existing_to_drop, axis=1, inplace=True)

# 9. Train/Test Split
# CRITICAL FIX: You must drop 'target' from X, or the model sees the answer!
features = df.drop(['parcelid', 'target'], axis=1, errors='ignore')
target = df['target'].values

X_train, X_val, Y_train, Y_val = train_test_split(
    features, target, test_size=0.1, random_state=22
)

print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")

# 10. Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 11. Model Training
models = [
    LinearRegression(), 
    XGBRegressor(verbosity=0), # verbosity=0 silences warnings
    Lasso(), 
    RandomForestRegressor(), 
    Ridge()
]

for model in models:
    model.fit(X_train, Y_train)
    
    print(f'{model.__class__.__name__} : ')
    
    train_preds = model.predict(X_train)
    print('Training Error (MAE): ', mae(Y_train, train_preds))
    
    val_preds = model.predict(X_val)
    print('Validation Error (MAE): ', mae(Y_val, val_preds))
    print('-'*30)
