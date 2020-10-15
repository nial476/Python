import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

sns.set()

directory = Path(__file__).resolve().parents[0]  # returns a path object with path current path
directory = directory/'Data'/'USA_Housing.csv'

pd.set_option('display.max_columns', None)  # show all columns of the dataframe in output

df = pd.read_csv(directory)
print(df.head())

print(df.info())

print(df.describe())

sns.pairplot(df)
plt.tight_layout()
plt.show()

sns.distplot(df['Price'])
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
model = LinearRegression()

model.fit(X_train, y_train)
print('beta[0]= ', model.intercept_)

coeff = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coeff)

prediction = model.predict(X_test)
plt.scatter(prediction, y_test)
plt.show()

sns.distplot(y_test-prediction)
plt.show()

print('mean absolute error = ', metrics.mean_absolute_error(y_test, prediction))
print('mean squared error = ', metrics.mean_squared_error(y_test, prediction))
print('root mean squared error = ', np.sqrt(metrics.mean_squared_error(y_test, prediction)))