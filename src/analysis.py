import pandas as pd
import gzip
import random

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

    print len(df)

    sampled_reviews = random.sample(df.keys(), 10000)
    df = {k: df[k] for k in df.keys() if k in sampled_reviews}
    print len(df)
    return pd.DataFrame.from_dict(df, orient='index')


if __name__ == '__main__':
    PATH = '../data/reviews_CDs_and_Vinyl.json.gz'
    df = getDF(PATH)