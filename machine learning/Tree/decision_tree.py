import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

directory = Path(__file__).resolve().parents[0]
pd.set_option('display.max_columns', None)
sns.set()

df = pd.read_csv(directory/'kyphosis.csv')
print(df.head())
print(df.info())

sns.pairplot(df, hue='Kyphosis', diag_kind='hist')
plt.show()

X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = DecisionTreeClassifier()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)

print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))