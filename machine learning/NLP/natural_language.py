import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# nltk.download_shell()

sns.set()
pd.set_option('display.max_columns', None)

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', names=['label', 'message'], sep='\t')
print(messages.head())

print(messages.groupby(by='label').describe())

messages['length'] = messages['message'].apply(len)
print(messages.head())

sns.distplot(messages['length'], bins=100, kde=False)
plt.show()

print(messages['length'].describe())
print(messages[messages['length'] == 910]['message'].iloc[0])

g = sns.FacetGrid(data=messages, col='label', sharey=False)
g.map(sns.distplot, 'length', kde=False)
plt.show()


def text_processing(mess):
    """
    1. remove punctuations
    2. join letters to get orginal string without punctuation
    3.remove stopwords
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


print(messages['message'].head().apply(text_processing))

bow_transformer = CountVectorizer(analyzer=text_processing)
bow_transformer.fit(messages['message'])
print(len(bow_transformer.vocabulary_))

messages_bow = bow_transformer.transform(messages['message'])
print('Shape of sparse matrix : ', messages_bow.shape)
print('Number of non zero occurances: ', messages_bow.nnz)

print('Sparsity of the sparse matrix = {}'.format(messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))

tfidf_transformer = TfidfTransformer().fit(messages_bow)
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

messages_tfidf = tfidf_transformer.transform(messages_bow)

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)
print(msg_train.head())

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_processing)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())])

pipeline.fit(msg_train, label_train)
prediction = pipeline.predict(msg_test)

print('confusion matrix = {} \n'.format(confusion_matrix(label_test, prediction)))
print('classification report \n ', classification_report(label_test, prediction))
