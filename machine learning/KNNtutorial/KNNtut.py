import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

sns.set()
pd.set_option('display.max_columns', None)
directory = Path(__file__).resolve().parents[0]

df = pd.read_csv(directory/'Classified Data', index_col=0)
print(df.head())

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_feature = scaler.transform(df.drop('TARGET CLASS', axis=1))

'''
takes all columns except the target column of df dataframe for 
new dataframe df_feature containing the scaled feature values
'''
df_feature = pd.DataFrame(scaled_feature, columns=df.columns[:-1])
print(df_feature.head())

X = df_feature
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, linestyle='--', color='green', marker='o', markerfacecolor='red', markersize='10')
plt.title('Error Value vs K')
plt.xlabel('K')
plt.ylabel('Error')
plt.tight_layout()
plt.show()

model = KNeighborsClassifier(n_neighbors=17)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))