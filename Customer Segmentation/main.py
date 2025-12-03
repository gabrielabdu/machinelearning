import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import math

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('new.csv')

print("Shape:", df.shape)
print(df.info())
print(df.describe().T)

for col in df.columns:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        print(f'Column {col} contains {null_count} null values.')

df = df.dropna()
print("Total values in the dataset after removing the null values:", len(df))

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
df["day"] = df["Dt_Customer"].dt.day
df["month"] = df["Dt_Customer"].dt.month
df["year"] = df["Dt_Customer"].dt.year

df.drop(['Z_CostContact', 'Z_Revenue', 'Dt_Customer'], axis=1, inplace=True)

objects = df.select_dtypes(include='object').columns.tolist()
floats = df.select_dtypes(include=['float', 'int']).columns.tolist()

print("Object columns:", objects)
print("Numerical columns:", floats)

plt.figure(figsize=(15, 10))
rows = math.ceil(len(objects) / 2)
for i, col in enumerate(objects):
    plt.subplot(rows, 2, i + 1)
    sb.countplot(x=df[col])
plt.tight_layout()
plt.show()

if 'Marital_Status' in df.columns:
    print(df['Marital_Status'].value_counts())

plt.figure(figsize=(15, 10))
for i, col in enumerate(objects):
    plt.subplot(rows, 2, i + 1)
    df_melted = df.melt(id_vars=[col], value_vars=['Response'], var_name='hue')
    sb.countplot(x=col, hue='value', data=df_melted)
plt.tight_layout()
plt.show()

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == object:
        df[col] = le.fit_transform(df[col])

plt.figure(figsize=(15, 15))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

scaler = StandardScaler()
data = scaler.fit_transform(df)

model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(data)

plt.figure(figsize=(7, 7))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
plt.show()

error = []
k_range = range(1, 21)
for n_clusters in k_range:
    model = KMeans(init='k-means++',
                   n_clusters=n_clusters,
                   max_iter=500,
                   random_state=22)
    model.fit(data)
    error.append(model.inertia_)

plt.figure(figsize=(10, 5))
sb.lineplot(x=k_range, y=error)
sb.scatterplot(x=k_range, y=error)
plt.show()

model = KMeans(init='k-means++',
               n_clusters=5,
               max_iter=500,
               random_state=22)
segments = model.fit_predict(data)

plt.figure(figsize=(7, 7))
df_tsne = pd.DataFrame({'x': tsne_data[:, 0], 'y': tsne_data[:, 1], 'segment': segments})
sb.scatterplot(x='x', y='y', hue='segment', data=df_tsne, palette='viridis')
plt.show()