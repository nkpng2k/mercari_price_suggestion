import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV

# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class MercariModeling(object):

    def __init__(self, filepath, delimiter=','):
        self.train = pd.read_csv(filepath, delimiter=delimiter)

    def nlp_vectorize(self, column_name):
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
