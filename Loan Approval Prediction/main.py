import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

data = pd.read_csv("LoanApprovalPrediction.csv")

data.drop(['Loan_ID'], axis=1, inplace=True)

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].mean())

obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)

plt.figure(figsize=(18, 36))
index = 1
for col in object_cols:
    y = data[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1
plt.show()

label_encoder = preprocessing.LabelEncoder()
for col in object_cols:
    data[col] = label_encoder.fit_transform(data[col])

plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.show()

sns.catplot(x="Gender", y="Married", hue="Loan_Status", kind="bar", data=data)
plt.show()

X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
svc = SVC()
lc = LogisticRegression(max_iter=1000)

for clf in (rfc, knn, svc, lc):
    clf.fit(X_train, Y_train)
    
    Y_pred_train = clf.predict(X_train)
    print("Accuracy score of ", clf.__class__.__name__, "(Train) =", 100 * metrics.accuracy_score(Y_train, Y_pred_train))

    Y_pred_test = clf.predict(X_test)
    print("Accuracy score of ", clf.__class__.__name__, "(Test) =", 100 * metrics.accuracy_score(Y_test, Y_pred_test))
