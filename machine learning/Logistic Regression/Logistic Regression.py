import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sns.set_style('whitegrid')

pd.set_option('display.max_columns', None)

directory = Path(__file__).resolve().parents[0]
directory = directory/'titanic'
train = pd.read_csv(directory/'train.csv')

print(train.head())
print(train.info())

sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
plt.show()

sns.countplot(x='Survived', data=train, hue='Sex')
plt.show()

sns.countplot(x='Survived', data=train, hue='Pclass')
plt.show()

sns.distplot(train['Age'].dropna(), bins=30)
plt.show()

sns.countplot(x='SibSp', data=train)
plt.show()

sns.distplot(train['Fare'], bins=40, kde=False)
plt.show()

plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Age', data=train)
plt.show()

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

sns.heatmap(train.isnull(), cbar=False, yticklabels=False)
plt.show()

train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)
print(train.head())
sns.heatmap(train.isnull(), yticklabels=False, cbar=True)
plt.show()

sex = pd.get_dummies(train['Sex'], drop_first=True)
print(sex.head())

embark = pd.get_dummies(train['Embarked'], drop_first=True)
print(embark.head())

pd.concat([train, sex, embark], axis=1)
print(train.head())

train.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
print(train.head())

X = train.drop('Survived', axis=1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

model = LogisticRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
print(classification_report(y_test, prediction))

print(confusion_matrix(y_test, prediction))