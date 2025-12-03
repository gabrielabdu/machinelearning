import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("ad_10000records.csv")
print(data.head())

print(data["Clicked on Ad"].value_counts())

click_through_rate = (data["Clicked on Ad"].sum() / len(data)) * 100
print(f"The click through rate is: {click_through_rate}%")

le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])

X = data.iloc[:, 0:7]
X = X.drop(['Ad Topic Line', 'City'], axis=1)

y = data["Clicked on Ad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("The model accuracy is", accuracy_score(y_test, y_pred))