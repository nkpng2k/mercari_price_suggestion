import pandas as pd
import nltk
from nltk.stem.wordnet import WordNetLemmatizer as wnl
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from string import punctuation


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

    def apply_func(self, new_name, from_col, func):
        self.train[new_name] = self.train[from_col].apply(lambda x: func(x))

    def engineer_features(self, csv_out_path):
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

    testing = pd.read_csv('data/new_features_added.csv')
    testing.head()
