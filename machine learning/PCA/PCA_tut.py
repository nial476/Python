import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sns.set()

cancer = load_breast_cancer()
print(type(cancer))
print(cancer.keys())
print(cancer['DESCR'])

df = pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'])
print(df.head())

scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

plt.figure(figsize=(10,8))
plt.scatter(x_pca[:,0], x_pca[:,1], c=cancer['target'], cmap='plasma')
plt.xlabel('first component')
plt.ylabel('second component')
plt.show()