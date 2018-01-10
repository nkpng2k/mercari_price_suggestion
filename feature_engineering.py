import pandas as pd
import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer as wnl
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from string import punctuation
from sklearn.metrics.pairwise import pairwise_distances as pw_dist
from sklearn.ensemble import RandomForestRegressor
from slkearn.metrics import jaccard_similarity_score as jss


class MercariFeatureEngineering(object):

    def __init__(self, filepath, delimiter=','):
        self.train = pd.read_csv(filepath, delimiter=delimiter)
        self.stop_words = set(stopwords.words('english'))
        self.alphabet = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                             'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                             'u', 'v', 'w', 'x', 'y', 'z'])
        self.rf = None

    def _get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def _which_tree_leaf(self, X):
        ret_mat = np.empty((X.shape[0], len(self.rf.estimators_)))
        for i, tree in enumerate(self.rf.estimators_):
            labels = tree.apply(X)
            ret_mat[:, i] = labels
        return ret_mat

    def _most_similar(self, similarity_matrix, n_similar):
        idx_top_sim = np.empty((similarity_matrix.shape[0], n_similar))
        for i, row in enumerate(similarity_matrix):
            top_sim = row.argsort()[-n_similar:][::-1]
            idx_top_sim[i] = top_sim
        return idx_top_sim

    def _jaccard_similarity(self, leaf_mat):
        similarity_matrix = 1 - pw_dist(leaf_mat, metric='jaccard')
        return similarity_matrix

    def _bad_jaccard(self, leaf_mat):
        sim_mat = np.empty((leaf_mat.shape[0], leaf_mat.shape[0]))
        count = 0
        for i in xrange(leaf_mat.shape[0]):
            print "top loop: {} completed".format(count)
            for j in xrange(leaf_mat.shape[0]):
                sim_mat[i, j] = jss(leaf_mat[i], leaf_mat[j])
            count += 1
        return sim_mat

    def fill_na(self, column_name, new_col, fill_with):
        self.train[new_col] = self.train[column_name].isnull().astype(int)
        self.train[column_name] = self.train[column_name].fillna(fill_with)

    def split_categories(self, column_name, split_on):
        top, middle, bottom = [], [], []
        for i, row in self.train.iterrows():
            hierarchy_string = row[column_name]
            hierarchy_list = hierarchy_string.split(split_on)
            top.append(hierarchy_list[0])
            middle.append(hierarchy_list[1])
            bottom.append(hierarchy_list[2])
        self.train['category_top'] = top
        self.train['category_middle'] = middle
        self.train['category_bottom'] = bottom

    def tokenize(self, string):
        clean_list = []
        description = string.lower()
        desc_list = description.split()
        for word in desc_list:
            word = word.strip(punctuation)
            if len(word) > 0:
                clean_list.append(word)
        return clean_list

    def no_stopwords(self, token_list):
        no_stop_words = [word for word in token_list
                         if word not in self.stop_words]
        return no_stop_words

    def lemmatize(self, token_list):
        lemmatizer = wnl()
        lemmed_tokens = []
        tagged = nltk.pos_tag(token_list)
        for word, pos_tag in tagged:
            word = ''.join([letter for letter in word
                            if letter in self.alphabet])
            lem_word = lemmatizer.lemmatize(word,
                                            pos=self._get_wordnet_pos(pos_tag))
            lemmed_tokens.append(lem_word)
        return lemmed_tokens

    def randomforest_similarity(self, n_estimators, X, y):
        self.rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1,
                                        verbose=5)
        self.rf.fit(X, y)
        print ('Done Fitting Forest')
        leaf_mat = self._which_tree_leaf(X)
        print ('Done Finding Which Leaf')
        sim_mat = self._bad_jaccard(leaf_mat)
        print ('Done Finding Similarity')
        return sim_mat

    def apply_func(self, new_name, from_col, func):
        self.train[new_name] = self.train[from_col].apply(lambda x: func(x))

    def drop_rows_with_value(self, column, value):
        self.train = self.train[self.train[column] != value]

    def engineer_features(self, csv_out_path):
        if 'price' in self.train.columns:
            self.drop_rows_with_value('price', 0)
        self.fill_na('category_name', 'cat_Was_null', 'None/None/None')
        self.fill_na('brand_name', 'brand_was_null', 'no_label')
        self.fill_na('item_description', 'desc_was_null', 'No description')
        print ('All Nulls Filled, New Binary Columns Created!')
        self.split_categories('category_name', '/')
        print('Categories Split!')
        self.apply_func('desc_tokens', 'item_description', self.tokenize)
        print('Tokenized!')
        self.apply_func('desc_tokens', 'desc_tokens', self.no_stopwords)
        print('No Stop Words!')
        self.apply_func('lemmed_tokens', 'desc_tokens', self.lemmatize)
        print('Lemmatized!')
        self.train.to_csv(csv_out_path, index=False)


if __name__ == "__main__":

    feat_eng = MercariFeatureEngineering('data/train.tsv', '\t')
    feat_eng.engineer_features('data/new_features_added.csv')
    feat_eng.train.to_csv('data/new_features_added.csv', index=False)
    feat_eng.train.head()

    testing = pd.read_csv('data/new_features_dropped_zeros.csv')
