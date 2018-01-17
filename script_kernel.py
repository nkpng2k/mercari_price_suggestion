import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import pairwise_distances as pw_dist
from datetime import datetime


class MercariFeatureEngineering(object):

    def __init__(self, train_filepath, test_filepath, delimiter=','):
        self.train_df = pd.read_csv(train_filepath, delimiter=delimiter)
        self.test_df = pd.read_csv(test_filepath, delimiter=delimiter)
        # self.stop_words = set(stopwords.words('english'))
        self.alphabet = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                             'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                             'u', 'v', 'w', 'x', 'y', 'z'])
        self.rf = None
        self.brand_encoder = LabelEncoder()
        self.cat_top_encoder = LabelEncoder()
        self.cat_mid_encoder = LabelEncoder()
        self.cat_bot_encoder = LabelEncoder()

    def fill_na(self, df, column_name, new_col, fill_with):
        df[new_col] = df[column_name].isnull().astype(int)
        df[column_name] = df[column_name].fillna(fill_with)

    def split_categories(self, df, column_name, split_on):
        top, middle, bottom = [], [], []
        for i, row in df.iterrows():
            hierarchy_string = row[column_name]
            hierarchy_list = hierarchy_string.split(split_on)
            top.append(hierarchy_list[0])
            middle.append(hierarchy_list[1])
            bottom.append(hierarchy_list[2])
        df['cat_top'] = top
        df['cat_mid'] = middle
        df['cat_bot'] = bottom

    def clean_column(self, brand_name):
        word = brand_name.lower()
        word = ''.join([letter for letter in word if letter in self.alphabet])
        return word

    def drop_rows_with_value(self, df, column, value):
        df = df[df[column] != value]

    def apply_func(self, df, new_name, from_col, func):
        df[new_name] = df[from_col].apply(lambda x: func(x))

    def _categorical_labels(self, df, column_name, new_col, encoder):
        df[new_col] = encoder.transform(df[column_name])

    def dummify_categories(self, col_top, col_mid, col_bot, col_brand, df):
        brand_dummies = pd.get_dummies(df[col_brand])
        cat_top_dummies = pd.get_dummies(df[col_top])
        cat_mid_dummies = pd.get_dummies(df[col_mid])
        cat_bot_dummies = pd.get_dummies(df[col_bot])
        result = pd.concat([brand_dummies, cat_top_dummies,
                           cat_mid_dummies, cat_bot_dummies],
                           axis=1, join_axes=[brand_dummies.index])
        return result

    def train_encoders(self):
        train = set(self.train_df['brand_name'])
        test = set(self.test_df['brand_name'])
        self.brand_encoder.fit(list(train.union(set(test))))

        train = set(self.train_df['cat_top'])
        test = set(self.test_df['cat_top'])
        self.cat_top_encoder.fit(list(train.union(set(test))))

        train = set(self.train_df['cat_mid'])
        test = set(self.test_df['cat_mid'])
        self.cat_mid_encoder.fit(list(train.union(set(test))))

        train = set(self.train_df['cat_bot'])
        test = set(self.test_df['cat_bot'])
        self.cat_bot_encoder.fit(list(train.union(set(test))))

    def create_labels(self, df):
        self._categorical_labels(df, 'brand_name',
                                 'brand_numeric', self.brand_encoder)
        self._categorical_labels(df, 'cat_top',
                                 'cat_top_numeric', self.cat_top_encoder)
        self._categorical_labels(df, 'cat_mid',
                                 'cat_mid_numeric', self.cat_mid_encoder)
        self._categorical_labels(df, 'cat_bot',
                                 'cat_bot_numeric', self.cat_bot_encoder)

    def engineer_features(self, df):
        if 'price' in df.columns:
            self.drop_rows_with_value(df, 'price', 0)
        self.fill_na(df, 'category_name', 'cat_Was_null', 'None/None/None')
        self.fill_na(df, 'brand_name', 'brand_was_null', 'no_label')
        self.fill_na(df, 'item_description', 'desc_was_null', 'No description')
        print('All Nulls Filled, New Binary Columns Created!')
        self.split_categories(df, 'category_name', '/')
        print('Categories Split!')
        self.apply_func(df, 'brand_name', 'brand_name', self.clean_column)
        self.apply_func(df, 'cat_top', 'cat_top', self.clean_column)
        self.apply_func(df, 'cat_mid', 'cat_mid', self.clean_column)
        self.apply_func(df, 'cat_bot', 'cat_bot', self.clean_column)
        print('Cleaned Columns')


class MercariRegression(object):

    def __init__(self):
        self.sim_rf = None

    def generator(self, X, subset_size=10):
        n_rows = X.shape[0]
        last = n_rows % subset_size
        n_iters = (n_rows - n_rows % subset_size)/subset_size
        for i in xrange(n_iters):
            yield X[(i*subset_size):(i+1)*subset_size]
        yield X[-last:]

    def which_leaf(self, X):
        ret_mat = np.empty((X.shape[0], len(self.sim_rf.estimators_)))
        for i, tree in enumerate(self.sim_rf.estimators_):
            labels = tree.apply(X)
            ret_mat[:, i] = labels
        return ret_mat

    def jac_sim(self, train_leaf, test_leaf):
        similarity_matrix = 1 - pw_dist(test_leaf, train_leaf,
                                        n_jobs=-1, metric='hamming')
        return similarity_matrix

    def top_n_similar(self, sim_mat, n_similar):
        idx_top_sim = np.empty((sim_mat.shape[0], n_similar))
        for i, row in enumerate(sim_mat):
            top_sim = row.argsort()[-n_similar:][::-1]
            idx_top_sim[i] = top_sim
            return idx_top_sim

    def avg_top_sim(self, df, idx_top_sim):
        avg_list = []
        for row in idx_top_sim:
            avg_price = np.mean(df.iloc[row]['price'])
            avg_list.append(avg_price)
        return avg_list

    def rf_sim(self, n_estimators, X_train, y_train, X_test):
        self.sim_rf = RandomForestRegressor(n_estimators=n_estimators,
                                            verbose=5, n_jobs=-1)
        self.sim_rf.fit(X_train, y_train)
        print ('Done Fitting Random Forest')


if __name__ == "__main__":
    start = datetime.now()
    f_eng = MercariFeatureEngineering('data/train.tsv', 'data/test.tsv',
                                      delimiter='\t')
    f_eng.engineer_features(f_eng.train_df)
    f_eng.engineer_features(f_eng.test_df)
    end = datetime.now()
    print (end-start)

    start = datetime.now()
    f_eng.train_encoders()
    f_eng.create_labels(f_eng.train_df)
    f_eng.create_labels(f_eng.test_df)
    end = datetime.now()
    print('Created Categorical Labels: {}'.format(end - start))

    start = datetime.now()
    X_train = f_eng.train_df[['brand_numeric', 'cat_top_numeric',
                             'cat_mid_numeric', 'cat_bot_numeric']]
    y_train = f_eng.train_df['price']
    X_test = f_eng.test_df[['brand_numeric', 'cat_top_numeric',
                           'cat_mid_numeric', 'cat_bot_numeric']]
    end = datetime.now()
    print (end - start)

    start = datetime.now()
    reg = MercariRegression()
    reg.rf_sim(100, X_train, y_train, X_test)
    end = datetime.now()
    print (end - start)

    start = datetime.now()
    leaves_df = f_eng.train_df[['brand_numeric', 'cat_top_numeric',
                                'cat_mid_numeric', 'cat_bot_numeric',
                                'price']].sample(frac=0.25)
    X_leaves = leaves_df.drop('price', axis=1)
    train_leaf = reg.which_leaf(X_leaves)
    mid = datetime.now()
    print ('Leaves for Train Set Found: {}'.format(mid-start))
    test_leaf = reg.which_leaf(X_test)
    end = datetime.now()
    print ("Leaves for Test Set Found: {}".format(end - mid))
    print ('Total Time: {}'.format(end - start))

    count = 0
    i = 0
    all_avgs = []
    start = datetime.now()
    for item in reg.generator(test_leaf, 1000):
        beg = datetime.now()
        count += 1
        sim = reg.jac_sim(train_leaf, item)
        idx_top_sim = reg.top_n_similar(sim, 50)
        avg_top_sim = reg.avg_top_sim(leaves_df, idx_top_sim)
        all_avgs.extend(avg_top_sim)
        after = datetime.now()
        print ('Loop: {}, Time: {}'.format(count, (after - beg)))
    end = datetime.now()
    print (end - start)
