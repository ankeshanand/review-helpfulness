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
inquirer_loc = 'data/inquirerbasic.csv'
top_100 = set([])

class LIWC(object):
    def __init__(self):
        self.words = []
        self.prefixes = []
        self.dim_map = {}
        self.dic = {}
        self.path = 'data/LIWC2007.dic'
        self.start_line = 66
        self.create_LIWC_dim_maps()
        self.compute_prefixes_and_words()
        self.create_dictionary()

    def create_LIWC_dim_maps(self):
        dim_idx = 0
        self.dim_map = {}
        with open(self.path) as f:
            for i, l in enumerate(f):
                if i == 0:
                    continue
                l = l.strip()
                if l == '%':
                    break
                given_dim = l.split('\t')[0]
                self.dim_map[given_dim] = dim_idx
                dim_idx += 1

    def compute_prefixes_and_words(self):
        with open(self.path) as f:
            for i, l in enumerate(f):
                if i < self.start_line:
                    continue
                target = l.strip().split('\t')[0]
                if target.endswith('*'):
                    self.prefixes.append(target[:-1])
                else:
                    self.words.append(target)

    def create_dictionary(self):
        with open(self.path) as f:
            for i, l in enumerate(f):
                if i < self.start_line:
                    continue
                l = l.strip().split('\t')
                word, dim = l[0], l[1:]

                if word.endswith('*'):
                    word = word[:-1]

                actual_dims = []
                for d in dim:
                    try:
                        actual_dims.append(self.dim_map[d])
                    except:
                        print l
                        print word
                        print dim
                self.dic[word] = actual_dims

def get_inquirer_df(inquirer_loc):
    df = pd.read_csv(inquirer_loc, dtype=str)
    df = df.drop(['Source', 'Othtags', 'Defined'], axis=1)
    return df

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
        if d['asin'] not in top_100:
            continue
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def get_top_100_products(path):
    top_100 = set([])
    review_counts = {}
    for d in parse(path):
        review_counts[d['asin']] = review_counts.get(d['asin'], 0) + 1
    for k in sorted(review_counts, key=review_counts.get, reverse=True)[:100]:
        top_100.add(k)
    print len(top_100)
    return top_100

def compute_score(votes):
    n, d = votes
    helpfulness = ( (1.0*n) / (1.0*d) )
    return helpfulness

#top_100 = get_top_100_products(data_loc)
#reviews = getDF(data_loc)
reviews = pd.read_pickle('../LSTM/reviews.pkl')
#pickle.dump(reviews, open('reviews_Electronics.p', 'wb'))
print 'Parsing complete.'
print len(reviews)

reviews['word_list']=reviews['reviewText'].apply(review_to_words)
reviews['scores']=reviews['helpful'].apply(compute_score)
print reviews['scores'].head(n=10)
y = reviews['scores']
Text = reviews['word_list']

del reviews

l = LIWC()
LIWC_dic = l.dic
LIWC_prefixes = l.prefixes
LIWC_words = l.words
print dict(LIWC_dic.items()[0:5])

df = get_inquirer_df(inquirer_loc)
inquirer_columns = list(df.columns[1:])
INQUIRER_MATRIX = np.zeros((len(Text), len(inquirer_columns)))
inquirer_words = {}

for i, item in enumerate(df.Entry.values):
    inquirer_words[item] = i

count = 0
for idx, review in enumerate(Text):
    if count % 200 == 0:
        print count
    count += 1
    for word in review:
        item = ''
        if word.upper() in inquirer_words:
            item = word.upper()
        elif word.upper() + '#1' in inquirer_words:
            item = word.upper() + '#1'
        else:
            continue

        row = df.ix[inquirer_words[item]]
        for i, col in enumerate(inquirer_columns):
            if not pd.isnull(row[col]):
                INQUIRER_MATRIX[idx][i] += 1

print 'Computed Inquirer Matrix'

X = np.zeros((len(Text), len(l.dim_map)))

for idx, review in enumerate(Text):
    for word in review:
        if word in LIWC_dic:
            for d in LIWC_dic[word]:
                X[idx][d] += 1
        else:
            if word.startswith(tuple(LIWC_prefixes)):
                for item in LIWC_prefixes:
                    if word.startswith(item):
                        for d in LIWC_dic[item]:
                            X[idx][d] += 1

X = np.hstack((X, INQUIRER_MATRIX))

print 'Computed X'
X = StandardScaler().fit_transform(X)
print X[0]

model = SVR()
params = {'C': [0.1, 0.5]}
grid = GridSearchCV(model, params, cv=5, scoring='mean_squared_error', n_jobs=-1)
grid.fit(X, y)
print grid.best_score_
print 'RMSE: ' + str(sqrt(abs(grid.best_score_)))
