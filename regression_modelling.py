import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import jaccard_similarity_score as jaccard

# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class MercariModeling(object):

    def __init__(self, filepath, delimiter=','):
        self.train = pd.read_csv(filepath, delimiter=delimiter)
        self.rf = None

    def _which_tree_leaf(self, X):
        ret_mat = np.empty((X.shape[0], len(self.rf.estimators_)))
        for i, tree in enumerate(self.rf.estimators_):
            labels = tree.apply(X)
            ret_mat[:, i] = labels
        return ret_mat

    def _most_similar(self, similarity_matrix, n_similar):
        idx_top_sim = np.empty((similarity_matrix.shape[0], n_similar))
        for i, row in enumerate(similarity_matrix):
            top_sim = similarity_matrix.argsort()[-n_similar:][::-1]
            idx_top_sim[i] = top_sim
        return idx_top_sim

    def _jaccard_similarity(self, leaf_mat):
        similarity_matrix = np.empty((leaf_mat.shape[0], leaf_mat.shape[0]))
        for i in xrange(leaf_mat.shape[0]):
            for j in xrange(leaf_mat.shape[0]):
                similarity_matrix[i, j] = jaccard(leaf_mat[i], leaf_mat[j])
        return similarity_matrix

    def nlp_vectorize(self, column_name):
        vectorizer = TfidfVectorizer().fit(self.train[column_name])
        # NOTE: removed preprocessor = lambda x: x, tokenizer = lambda x: x
        #       It was giving weird results
        vect_tokens = vectorizer.transform(self.train[column_name])

        return vectorizer, vect_tokens

    def reduce_dimensions(self, vectorized_tokens):
        trunc_svd = TruncatedSVD(n_components=150)
        trunc_svd.fit(vectorized_tokens)
        svd_tokens = trunc_svd.transform(vectorized_tokens)

        return trunc_svd, svd_tokens

    def randomforest_similarity(self, n_estimators, X, y):
        self.rf = RandomForestRegressor(n_estimators=n_estimators)
        self.rf.fit(X, y)
        print ('Done Fitting Forest')
        leaf_mat = self._which_tree_leaf(X)
        print ('Done Finding Which Leaf')
        sim_mat = self._jaccard_similarity(leaf_mat)
        print ('Done Finding Similarity')
        return sim_mat

    def create_test_train_splits(self, train_columns, target_column, split):
        if type(train_columns) is list:
            train_data = self.train(train_columns)
        else:
            train_data = train_columns

        if type(target_column) is str:
            target_data = self.train[target_column]
        else:
            target_data = target_column

        X_train, X_test, y_train, y_test = train_test_split(train_data,
                                                            target_data,
                                                            test_size=split)

        return X_train, X_test, y_train, y_test

    def gridsearch_model(self, model, parameters, X, y):
        clf = GridSearchCV(model, parameters,
                           scoring='neg_mean_squared_log_error',
                           n_jobs=-1, cv=5, verbose=5)
        clf.fit(X, y)

        return clf.best_estimator_, clf.best_params_


if __name__ == "__main__":
    models = MercariModeling('data/new_features_added.csv')
