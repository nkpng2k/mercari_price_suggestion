import pandas as pd
import nltk
from nltk.stem.wordnet import WordNetLemmatizer as wnl
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from string import punctuation
# from sklearn.decomposition import TruncatedSVD
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.model_selection import train_test_split


class MercariFeatureEngineering(object):

    def __init__(self, filepath, delimiter=','):
        self.train = pd.read_csv(filepath, delimiter=delimiter)
        self.stop_words = set(stopwords.words('english'))
        self.alphabet = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                             'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                             'u', 'v', 'w', 'x', 'y', 'z'])

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

    def remove_stopwords(self, token_list):
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

    def apply_func(self, new_name, from_col, func):
        self.train[new_name] = self.train[from_col].apply(lambda x: func(x))

    def engineer_features(self):
        self.fill_na('category_name', 'cat_Was_null', 'None/None/None')
        self.fill_na('brand_name', 'brand_was_null', 'no_label')
        self.fill_na('item_description', 'desc_was_null', 'No description')
        self.split_categories('category_name', '/')
        self.apply_func('desc_tokens', 'item_description', self.tokenize)
        self.apply_func('desc_tokens', 'desc_tokens', self.remove_stopwords)
        self.apply_func('lemmed_tokens', 'desc_tokens', self.lemmatize)


if __name__ == "__main__":

    feat_eng = MercariFeatureEngineering('data/train.tsv', '\t')
    feat_eng.engineer_features()
    test_eng = MercariFeatureEngineering('data/train.tsv', '\t')
    test_eng.fill_na('category_name', 'cat_was_null', 'None/None/None')
    test_eng.fill_na('brand_name', 'brand_Was_null', 'no_label')
    test_eng.fill_na('item_description', 'desc_was_null', 'No description')
    test_eng.split_categories('category_name', '/')

    test_eng.apply_func('desc_tokens', 'item_description', test_eng.tokenize)
    test_eng.apply_func('desc_tokens', 'desc_tokens', test_eng.remove_stopwords)
    test_eng.apply_func('lemmed_tokens', 'desc_tokens', test_eng.lemmatize)


    # stop_words = set(stopwords.words('english'))
    #
    # train = pd.read_csv('data/train.tsv', delimiter='\t')
    # train['category_name'] = train['category_name'].fillna('None/None/None')
    #
    # train['brand_name'] = train['brand_name'].fillna('no_label')
    #
    # train['item_description'] = train['item_description'].fillna('No description')
    #
    # top, middle, bottom = [], [], []
    # for i, row in train.iterrows():
    #     hierarchy_string = row['category_name']
    #     hierarchy_list = hierarchy_string.split('/')
    #     top.append(hierarchy_list[0])
    #     middle.append(hierarchy_list[1])
    #     bottom.append(hierarchy_list[2])
    # train['category_top'] = top
    # train['category_middle'] = middle
    # train['category_bottom'] = bottom
    #
    # train['description_tokens'] = train['item_description'].apply(lambda x: tokenize(x))
    #
    # train['description_tokens'] = train['description_tokens'].apply(lambda x: remove_stopwords(x))
    #
    # train['lemmed_tokens'] = train['description_tokens'].apply(lambda x: lemmatize(x))
