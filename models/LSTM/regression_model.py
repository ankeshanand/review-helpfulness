import re
import sqlite3
import pandas as pd
import numpy as np
import gzip
import random

from gensim.models import Word2Vec
from bs4 import BeautifulSoup
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.utils import np_utils, generic_utils
from keras import backend as K

from constants import DATA_LOC_CD, DATA_LOC_ELECTRONICS, DATA_LOC_SPORTS

import theano
theano.config.openmp = True


def root_mean_squared_error(y_true, y_pred):
    """
    RMSE loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def parse(path):
    with gzip.open(path, 'rb') as g:
        for l in g:
            yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        if 'helpful' not in d or len(d['helpful']) != 2 or d['helpful'][1] < 5:
            continue
        df[i] = d
        i += 1
    sampled_reviews = random.sample(df.keys(), 10000)
    df = {k: df[k] for k in df.keys() if k in sampled_reviews}
    return pd.DataFrame.from_dict(df, orient='index')


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

def compute_score(votes):
    n, d = votes
    helpfulness = ( (1.0*n) / (1.0*d) )
    return helpfulness

# Retrieve the reviews with more than 5 votes

#print 'Parsing CD Reviews'
#cd_reviews = getDF(DATA_LOC_CD)
#print 'Parsing Electronics Reviews'
#electronics_reviews = getDF(DATA_LOC_ELECTRONICS)
#print 'Parsing Sports Reviews'
#sports_reviews = getDF(DATA_LOC_SPORTS)

#reviews = pd.concat([cd_reviews, electronics_reviews, sports_reviews], ignore_index=True)
#print 'Parsing complete.'
#print len(reviews)
reviews = pd.read_pickle('reviews.pkl')

reviews['word_list']=reviews['reviewText'].apply(review_to_words)
reviews['scores'] = reviews['helpful'].apply(compute_score)
print reviews['scores'].head(n=10)
Text_train, Text_test, y_train, y_test = train_test_split(reviews['word_list'], reviews['scores'],
        test_size=0.2, random_state=20)
print y_train.shape

del reviews

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
model.add(LSTM(200, input_shape=(sequence_size, dimsize), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(200))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile(loss=root_mean_squared_error,
              optimizer='rmsprop')

print 'Training the LSTM model'
batch_size = 128
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10,
          validation_split=0.2)
score = model.evaluate(X_test, y_test,batch_size=batch_size)
print score

import logging
logname='results.log'
logging.basicConfig(filename=logname,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
logging.info('Score= ' + str(score) + ' Model: LSTM ' + ' Layers: 2 ' + ' HiddenLayerDimension: 200*200 ' + ' Dropout: 0.5' + ' ReLu act. ')
