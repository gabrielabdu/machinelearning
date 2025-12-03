import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as ras, classification_report, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('parkinson_disease.csv')

if 'id' in df.columns:
    df = df.groupby('id').mean().reset_index()
    df.drop('id', axis=1, inplace=True)

corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
df.drop(to_drop, axis=1, inplace=True)

X = df.drop('class', axis=1)
y = df['class']

X_norm = MinMaxScaler().fit_transform(X)
selector = SelectKBest(chi2, k=min(30, X.shape[1]))
selector.fit(X_norm, y)
X_selected = X.loc[:, selector.get_support()]

class_counts = y.value_counts()
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
plt.show()

X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=10)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

ros = RandomOverSampler(sampling_strategy=1.0, random_state=0)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

models = [
    LogisticRegression(class_weight='balanced'), 
    XGBClassifier(eval_metric='logloss'), 
    SVC(kernel='rbf', probability=True)
]

for model in models:
    model.fit(X_train_resampled, y_train_resampled)
    print(f'{model} : ')

    train_preds = model.predict(X_train_resampled)
    print('Training ROC AUC Score : ', ras(y_train_resampled, train_preds))

    val_preds = model.predict(X_val)
    print('Validation ROC AUC Score : ', ras(y_val, val_preds))
    print()

ConfusionMatrixDisplay.from_estimator(models[0], X_val, y_val)
plt.show()

print(classification_report(y_val, models[0].predict(X_val)))
