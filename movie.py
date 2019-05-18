#for importing and analysing data
import pandas as pd

#read data
input_data = pd.read_csv('User_movie_review.csv')

#preprocessing
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stemmer = PorterStemmer()
def tokenize(text):
    text = stemmer.stem(text)
    text = re.sub(r'\W+|\d+|_', ' ', text)
    tokens = nltk.word_tokenize(text)
    return tokens
    
#implement both tokenization and occurrence counting in a single class
from sklearn.feature_extraction.text import CountVectorizer

countvec = CountVectorizer(min_df = 5, tokenizer = tokenize, stop_words = stopwords.words('english'))
dtm = pd.DataFrame(countvec.fit_transform(input_data['text']).toarray(), columns = countvec.get_feature_names())

#adding label column
dtm['class'] = input_data['class']
dtm.head()

#building training and testing sets
df_train = dtm[:1900]
df_test = dtm[1900:]

#building naive bayes model
from sklearn.naive_bayes import MultinomialNB

#for classification with discrete features
clf = MultinomialNB()
x_train = df_train.drop(['class'], axis = 1)

#fitting model to our data
clf.fit(x_train, df_train['class'])

#accuracy
x_test = df_test.drop(['class'], axis = 1)
clf.score(x_test, df_test['class'])

#prediction
pred_sentiment = clf.predict(df_test.drop('class', axis = 1))
print(pred_sentiment)
