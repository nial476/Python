import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

sns.set()
pd.set_option('display.max_column', None)

cancer = load_breast_cancer()
print(cancer.keys())
print((cancer['DESCR']))

X = pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'])
y = cancer['target']
print(X.head())

plt.figure(figsize=(24,12))
sns.heatmap(X.corr(), annot=True)
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = SVC()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(estimator=SVC(), param_grid=param_grid, verbose=0)
grid.fit(X_train, y_train)

print(grid.best_estimator_)
print(grid.best_params_)
print(grid.best_score_)

pred = grid.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

