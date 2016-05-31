import re
import pandas as pd
import numpy as np
import gzip
from textstat.textstat import textstat as ts

from bs4 import BeautifulSoup
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from math import sqrt


def compute_score(votes):
    n, d = votes
    helpfulness = ( (1.0*n) / (1.0*d) )
    return helpfulness

import cPickle as pickle
reviews = pickle.load(open('reviews_Books.p', 'rb'))

print 'Parsing complete.'
print len(reviews)

reviews['scores'] = reviews['helpful'].apply(compute_score)
print reviews['scores'].head(n=10)
y = reviews['scores']
Text = reviews['reviewText']
del reviews

X = np.zeros((len(Text), 4))

for idx, review in enumerate(Text):
    if review == '':
        continue
    try:
        X[idx][0] = ts.flesch_reading_ease(review)
        X[idx][1] = ts.flesch_kincaid_grade(review)
        X[idx][2] = ts.gunning_fog(review)
        X[idx][3] = ts.smog_index(review)
    except Exception as e:
        print review
        print e

X = StandardScaler().fit_transform(X)
print 'Computed X'
print X[0]

model = SVR(verbose=True)
params = {'C': [0.1, 0.5]}
grid = GridSearchCV(model, params, cv=10, scoring='mean_squared_error', n_jobs=-1)
grid.fit(X, y)
print grid.best_score_
print 'RMSE: ' + str(sqrt(abs(grid.best_score_)))






