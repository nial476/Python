import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)
print(df.head())

movie_title = pd.read_csv('Movie_Id_Titles')
print(movie_title.head())

df = pd.merge(left=df, right=movie_title, how='inner', on='item_id')
print(df.head())

print(df.groupby(by='title', axis=0)['rating'].mean().sort_values(ascending=False))

print(df.groupby(by='title', axis=0)['rating'].count().sort_values(ascending=False))

rating = pd.DataFrame(data=df.groupby(by='title')['rating'].mean())
rating['num of rating'] = pd.DataFrame(data=df.groupby(by='title')['rating'].count())
print(rating.head())

rating['num of rating'].hist(bins=70)
plt.show()

sns.distplot(rating['rating'])
plt.show()

sns.jointplot(x='num of rating', y='rating', data=rating, height=10, alpha=0.5)
plt.show()

movie_mat = df.pivot_table(index='user_id', columns='title', values='rating')
print(movie_mat.head())

star_wars_rating = movie_mat['Star Wars (1977)']
liar_liar_rating = movie_mat['Liar Liar (1997)']

print(star_wars_rating.head())
print(liar_liar_rating.head())

similar_to_starwars = movie_mat.corrwith(other=star_wars_rating)
similar_to_liarliar = movie_mat.corrwith(other=liar_liar_rating)

corr_starwars = pd.DataFrame(data=similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)

corr_liarliar = pd.DataFrame(data=similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)

print(corr_starwars.head())
print(corr_liarliar.head())

corr_starwars = corr_starwars.join(other=rating['num of rating'])
corr_liarliar = corr_liarliar.join(other=rating['num of rating'])

print(corr_starwars)
print(corr_liarliar.head())

