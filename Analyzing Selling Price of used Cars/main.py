import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

df = pd.read_csv('output.csv')

df = df.iloc[:, 1:]

headers = ["symboling", "normalized-losses", "make", 
           "fuel-type", "aspiration", "num-of-doors",
           "body-style", "drive-wheels", "engine-location",
           "wheel-base", "length", "width", "height", "curb-weight",
           "engine-type", "num-of-cylinders", "engine-size", 
           "fuel-system", "bore", "stroke", "compression-ratio",
           "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

df.columns = headers

data = df.copy()

data.isna().any()
data.isnull().any()

data['city-mpg'] = 235 / data['city-mpg']
data.rename(columns={'city-mpg': "city-L / 100km"}, inplace=True)

print(data.columns)
print(data.dtypes)
print(data['price'].unique())

data = data[data['price'] != '?'].copy()
data['price'] = data['price'].astype(int)

print(data.dtypes)

data['length'] = data['length'] / data['length'].max()
data['width'] = data['width'] / data['width'].max()
data['height'] = data['height'] / data['height'].max()

bins = np.linspace(min(data['price']), max(data['price']), 4)
group_names = ['Low', 'Medium', 'High']
data['price-binned'] = pd.cut(data['price'], bins, labels=group_names, include_lowest=True)

print(data['price-binned'])

plt.figure()
plt.hist(data['price-binned'].astype(str))
plt.show()

print(pd.get_dummies(data['fuel-type']).head())
print(data.describe())

plt.figure()
plt.boxplot(data['price'])
plt.show()

plt.figure()
sns.boxplot(x='drive-wheels', y='price', data=data)
plt.show()

plt.figure()
plt.scatter(data['engine-size'], data['price'])
plt.title('Scatterplot of Enginesize vs Price')
plt.xlabel('Engine size')
plt.ylabel('Price')
plt.grid()
plt.show()

test = data[['drive-wheels', 'body-style', 'price']]
data_grp = test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()

print(data_grp)

data_pivot = data_grp.pivot(index='drive-wheels', columns='body-style', values='price')
print(data_pivot)

plt.figure()
plt.pcolor(data_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

data_annova = data[['make', 'price']]
grouped_annova = data_annova.groupby(['make'])
annova_results_l = stats.f_oneway(
    grouped_annova.get_group('honda')['price'],
    grouped_annova.get_group('subaru')['price']
)
print(annova_results_l)

plt.figure()
sns.regplot(x='engine-size', y='price', data=data)
plt.ylim(bottom=0)
plt.show()