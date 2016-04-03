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
    count_0, count_1, count_2 = 0,0,0
    labels = []
    for n,d in zip(nums,denms):
        category = None
        helpfulness = 100* ( (1.0*n) / (1.0*d) )
        if helpfulness <= 30:
            category = 0
            count_0 += 1
        elif helpfulness < 80:
            category = 1
            count_1 += 1
        else:
            category = 2
            count_2 += 1
        labels.append(category)
    print count_0, count_1, count_2
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
        test_size=0.2, random_state=42)


#size of hidden layer (length of continuous word representation)
dimsize=400

#train word2vec model
w2v = Word2Vec(reviews['word_list'].values, size=dimsize, window=5, min_count=1, workers=4)

# create training and test data matrix
sequence_size = 100
X_train = np.empty((len(Text_train), sequence_size, dimsize))
X_test = np.empty((len(Text_test), sequence_size, dimsize))

for review in Text_train:
    sequence = np.empty((sequence_size, dimsize))
    review = review[:sequence_size]
    tokens = review
    for i, token in enumerate(tokens):
        try:
            sequence[i] = w2v[token]
        except:
            print token
    X_train[i] = sequence

for review in Text_test:
    sequence = np.empty((sequence_size, dimsize))
    review = review[:sequence_size]
    tokens = review
    for i, token in enumerate(tokens):
        sequence[i] = w2v[token]
    X_test[i] = sequence

# build the keras LSTM model
model = Sequential()
model.add(LSTM(400, input_shape=(sequence_size, dimsize)))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam')

print 'Training the LSTM model'
batch_size = 32
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test), show_accuracy=True)
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size,
                            show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)




