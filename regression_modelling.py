import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class MercariModeling(object):

    def __init__(self, filepath, delimiter=','):
        self.train = pd.read_csv(filepath, delimiter=delimiter)

    def vectorize(self, column_name):
        vectorizer = TfidfVectorizer(preprocessor=lambda x: x,
                                     tokenizer=lambda x: x)
        vectorizer.fit(self.train[column_name])
        vect_tokens = vectorizer.transform(self.train[column_name])

        return vectorizer, vect_tokens

    def reduce_dimensions(self, vectorized_tokens):
        trunc_svd = TruncatedSVD(n_components=150)
        trunc_svd.fit(vectorized_tokens)
        svd_tokens = trunc_svd.transform(vectorized_tokens)

        return trunc_svd, svd_tokens
