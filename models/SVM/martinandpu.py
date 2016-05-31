import re
import pandas as pd
import numpy as np
import gzip

from bs4 import BeautifulSoup
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from math import sqrt

data_loc = 'data/reviews_Books.json.gz'
galc_loc = 'data/galc.csv'
top_100 = set([])

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

import cPickle as pickle
reviews = pickle.load(open('reviews_Books.p', 'rb'))

print 'Parsing complete.'
print len(reviews)

reviews['scores'] = reviews['helpful'].apply(compute_score)
reviews['word_list']=reviews['reviewText'].apply(review_to_words)
print reviews['scores'].head(n=10)
y = reviews['scores']
Text = reviews['word_list']
del reviews

def get_galc_dic(galc_loc):
    result = {}
    df = pd.read_csv(galc_loc,dtype=str)
    for i, row in df.iterrows():
        for item in row[1:]:
            if not pd.isnull(item):
                if item.endswith('*'):
                    item = item[:-1]
                result[item] = i
    return result

galc_dim = 38 + 1
galc_dic = get_galc_dic(galc_loc)
print galc_dic
print len(galc_dic)
prefixes = galc_dic.keys()
X = np.zeros((len(Text), galc_dim))

for idx, review in enumerate(Text):
    for word in review:
        if word.startswith(tuple(prefixes)):
            for item in prefixes:
                if word.startswith(item):
                    X[idx][galc_dic[item]] += 1
                    break
        else:
            X[idx][galc_dim-1] += 1

print 'Computed X'
X = StandardScaler().fit_transform(X)
print X[0]
print X[1]
print X[2]

model = SVR()
params = {'C': [0.1, 0.5]}
grid = GridSearchCV(model, params, cv=5, scoring='mean_squared_error', n_jobs=-1)
grid.fit(X, y)
print grid.best_score_
print 'RMSE: ' + str(sqrt(abs(grid.best_score_)))




