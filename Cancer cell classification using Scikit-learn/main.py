from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.sample(5))

print(df.info())

print(df.describe())

df2 = pd.DataFrame(data.target, columns=['target'])
print(df2.sample(5))

class_counts = df2["target"].value_counts()
labels = [data.target_names[i] for i in class_counts.index]

plt.pie(class_counts, labels=labels, autopct='%1.2f%%', colors=['red', 'green'])
plt.title("Breast Cancer Target Distribution")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred[:10])

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
