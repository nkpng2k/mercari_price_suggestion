import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from time import time
from feature_engineering import MercariFeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


def process_pricing_data(filename, csv_out_path):
    '''
    Returns a tuple containing:
        - a dataframe of tokenized item descriptions
        - a sparse matrix of cosine similarities between reduced item descriptions
    '''
    # Process Data
    MercariFeatureEngineering(
        filename, '\t').engineer_features(csv_out_path)
    pricing_contents = pd.read_csv(csv_out_path)
    # Tfidf on lemmatized item descriptions
    vectorizer = TfidfVectorizer(
        preprocessor=lambda x: x, tokenizer=lambda x: x).fit(pricing_contents['lemmed_tokens'])
    lem_vectorized = vectorizer.transform(pricing_contents['lemmed_tokens'])

    # Put into similarity matrix
    pricing_as_mat = cosine_similarity(lem_vectorized)
    return pricing_contents, pricing_as_mat


def x_most_similar(mat, item_idx, x=100, sim_thresh=0.98):
    '''
    INPUT:
        - matrix of cosine similarities
        - id for row calculating similarities
        - number of rows to compare that id to
    OUTPUT:
        - list of indices of x most similar rows
    '''
    # sim_sort = np.argsort(mat[item_id, :50])
    # print sim_sort
    mask = mat[item_idx, :] > sim_thresh
    top_idxs = np.ma.nonzero(mask)
    top_idxs = [x for idx in top_idxs for x in idx if x != item_idx]
    return top_idxs


def avg_similar_items(price_mat, similar_idxs):
    prices = []
    for idx in similar_idxs:
        prices.append(price_mat.iloc[idx]['price'])
    avg_price = np.mean(prices)
    return avg_price


def price_diff(price_mat, idx, avg_price):
    return price_mat.iloc[idx]['price'] - avg_price


if __name__ == "__main__":
    pricing_data_contents, pricing_mat = process_pricing_data(
        '/Users/hslord/kaggle/mercari_price_suggestion/data/train_sample.tsv',
        'data/new_features_added.csv')
    pricing_data_contents.iloc[877]
    pricing_data_contents.head()
    similar_1 = x_most_similar(pricing_mat, 877)
    avg_price_1 = avg_similar_items(pricing_data_contents, similar_1)
    print price_diff(pricing_data_contents, 877, avg_price_1)

    for x in similar_1:
        print pricing_data_contents.iloc[x]['item_description']
