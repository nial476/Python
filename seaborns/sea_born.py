import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.graph_objects as go

tips=sns.load_dataset('tips')
flights=sns.load_dataset('flights')
print(tips.head())
print(flights.head())

sns.distplot(tips['total_bill'], kde=False, bins=30)
plt.show()

sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')
plt.show()

sns.pairplot(tips, hue='sex', palette='coolwarm')
plt.show()

sns.rugplot(tips['total_bill'])
plt.show()

sns.barplot(x= 'total_bill', y= 'sex', data= tips, estimator=np.std)
plt.show()

sns.countplot(x='sex', data= tips)
plt.show()

sns.boxplot(x= 'sex', y= 'total_bill', data= tips, hue= 'smoker')
plt.show()

sns.violinplot(x= 'day', y= 'total_bill', data= tips, hue= 'sex', split= True )
plt.show()

sns.stripplot(x= 'day', y= 'total_bill', data= tips, jitter= True, hue= 'sex', split= True )
plt.show()

sns.swarmplot(x= 'day', y= 'total_bill', data= tips, color= 'black')
plt.show()

sns.factorplot(x= 'day', y='total_bill', data=tips, kind= 'swarm')
plt.show()

tc= tips.corr()
sns.heatmap(tc, annot=True)
plt.show()

fp=flights.pivot_table(index='month',columns='year',values='passengers')
sns.heatmap(fp, linecolor='white', linewidths=1)
plt.show()

sns.clustermap(fp)
plt.show()

g=sns.PairGrid(tips)
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
plt.show()

g=sns.FacetGrid(data=tips, col='day', row= 'smoker')
g.map(sns.distplot, 'total_bill')
plt.show()

sns.lmplot(x='total_bill', y='tip', data=tips, hue= 'sex', markers=['o', 'v'], col='day', row='smoker')
plt.show()


