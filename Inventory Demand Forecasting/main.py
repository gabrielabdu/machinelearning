import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error as mae
import holidays
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('StoreDemand.csv')
display(df.head())
display(df.tail())

print(df.shape)
print(df.info())
print(df.describe())

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

df['weekend'] = (df['weekday'] >= 5).astype(int)

india_holidays = holidays.IN()
df['holidays'] = df['date'].isin(india_holidays).astype(int)

df['m1'] = np.sin(df['month'] * (2 * np.pi / 12))
df['m2'] = np.cos(df['month'] * (2 * np.pi / 12))

df.drop('date', axis=1, inplace=True)

print(df['store'].nunique(), df['item'].nunique())

plot_features = ['store', 'year', 'month', 'weekday', 'weekend', 'holidays']

plt.subplots(figsize=(20, 10))
for i, col in enumerate(plot_features):
    plt.subplot(2, 3, i + 1)
    df.groupby(col)['sales'].mean().plot.bar()
plt.show()

plt.figure(figsize=(10,5))
df.groupby('day')['sales'].mean().plot()
plt.show()

plt.figure(figsize=(15, 10))
window_size = 30
data = df[df['year']==2013].copy()
windows = data['sales'].rolling(window_size)
sma = windows.mean()
sma = sma[window_size - 1:]

data['sales'].plot(label='Original')
sma.plot(label='SMA 30')
plt.legend()
plt.show()

plt.subplots(figsize=(12, 5))
plt.subplot(1, 2, 1)
sb.histplot(df['sales'], kde=True)

plt.subplot(1, 2, 2)
sb.boxplot(x=df['sales'])
plt.show()

plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

df = df[df['sales']<140]

X = df.drop(['sales', 'year'], axis=1)
y = df['sales'].values

X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.05, random_state=22)
print(X_train.shape, X_val.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

models = [LinearRegression(), XGBRegressor(), Lasso(), Ridge()]

for model in models:
    model.fit(X_train, Y_train)
    
    print(f'{model} : ')
    
    train_preds = model.predict(X_train)
    print('Training Error : ', mae(Y_train, train_preds))
    
    val_preds = model.predict(X_val)
    print('Validation Error : ', mae(Y_val, val_preds))
    print()
