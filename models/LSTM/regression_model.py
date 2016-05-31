import re
import sqlite3
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from bs4 import BeautifulSoup
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.utils import np_utils, generic_utils
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    """
    RMSE loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


data_loc= 'data/database.sqlite'

def review_to_words( review ):
    """
    Return a list of cleaned word tokens from the raw review

    """
    #Remove any HTML tags and convert to lower case
    review_text = BeautifulSoup(review).get_text().lower()

    #Replace smiliey and frown faces, ! and ? with coded word SM{int} in case these are valuable
    review_text=re.sub("(:\))",r' SM1',review_text)
    review_text=re.sub("(:\()",r' SM2',review_text)
    review_text=re.sub("(!)",r' SM3',review_text)
    review_text=re.sub("(\?)",r' SM4',review_text)

    #keep 'not' and the next word as negation may be important
    review_text=re.sub(r"not\s\b(.*?)\b", r"not_\1", review_text)

    #keep letters and the coded words above, replace the rest with whitespace
    nonnumbers_only=re.sub("[^a-zA-Z\_(SM\d)]"," ",review_text)

    #Split into individual words on whitespace
    words = nonnumbers_only.split()

    #Remove stop words
    words = [w for w in words]

    return (words)

def categorize_scores(nums, denms):
    labels = []
    for n,d in zip(nums,denms):
        helpfulness = ( (1.0*n) / (1.0*d) )
        labels.append(helpfulness)
    return labels

# Retrieve the reviews with more than 5 votes
connection = sqlite3.connect(data_loc)
reviews = pd.read_sql_query(""" SELECT Text, HelpfulnessNumerator, HelpfulnessDenominator
        FROM Reviews WHERE HelpfulnessDenominator >= 5""", connection)
reviews['word_list']=reviews['Text'].apply(review_to_words)
reviews['scores'] = categorize_scores(reviews['HelpfulnessNumerator'],
        reviews['HelpfulnessDenominator'])
print reviews['scores'].head(n=10)
Text_train, Text_test, y_train, y_test = train_test_split(reviews['word_list'], reviews['scores'],
        test_size=0.3, random_state=20)
print y_train.shape

#size of hidden layer (length of continuous word representation)
dimsize=300

#train word2vec model
w2v = Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)

# create training and test data matrix
sequence_size = 100
X_train = np.empty((len(Text_train), sequence_size, dimsize))
X_test = np.empty((len(Text_test), sequence_size, dimsize))

for idx, review in enumerate(Text_train):
    sequence = np.empty((sequence_size, dimsize))
    tokens = review
    count = 0
    for token in tokens:
        if count == 100:
            break
        try:
            token = token.lower()
            sequence[count] = w2v[token]
            count += 1
        except:
            pass
    X_train[idx] = sequence

for idx, review in enumerate(Text_test):
    sequence = np.empty((sequence_size, dimsize))
    tokens = review
    count = 0
    for token in tokens:
        if count == 100:
            break
        try:
            token = token.lower()
            sequence[count] = w2v[token]
            count += 1
        except:
            pass
    X_test[idx] = sequence

# build the keras LSTM model
model = Sequential()
model.add(LSTM(300, input_shape=(sequence_size, dimsize)))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss=root_mean_squared_error,
              optimizer='rmsprop')

print 'Training the LSTM model'
batch_size = 32
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5,
          validation_split=0.2)
score = model.evaluate(X_test, y_test,batch_size=batch_size)
print score



