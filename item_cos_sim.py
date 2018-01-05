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
        filename, ',').engineer_features(csv_out_path)
    pricing_contents = pd.read_csv(csv_out_path)
    # Tfidf on lemmatized item descriptions
    vectorizer = TfidfVectorizer(
        preprocessor=lambda x: x, tokenizer=lambda x: x).fit(pricing_contents['lemmed_tokens'])
    lem_vectorized = vectorizer.transform(pricing_contents['lemmed_tokens'])

    # Tfidf on brand names
    vectorizer = TfidfVectorizer().fit(pricing_contents['brand_name'])
    brand_vectorized = vectorizer.transform(pricing_contents['brand_name'])

    # Put into one matrix
    pricing_as_mat = sparse.hstack(
        (lem_vectorized, brand_vectorized))

    # Put into similarity matrix
    pricing_as_mat = cosine_similarity(pricing_as_mat)
    print 'Cosine Similarity Done!'
    return pricing_contents, pricing_as_mat


def x_most_similar(mat, item_idx, x=100, sim_thresh=0.65):
    '''
    INPUT:
        - matrix of cosine similarities
        - id for row calculating similarities
        - number of rows to compare that id to
    OUTPUT:
        - list of indices of x most similar rows
    '''
    top_idxs = np.argpartition(mat, -x)[item_idx, -x:]
    top_idxs = [idx for idx in top_idxs if idx != item_idx]
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
        '/Users/hslord/kaggle/mercari_price_suggestion/data/train_sample.csv',
        'data/new_features_added.csv')
    pricing_data_contents.iloc[118]
    pricing_data_contents.head()
    similar_1 = x_most_similar(pricing_mat, 118)
    avg_price_1 = avg_similar_items(pricing_data_contents, similar_1)
    print price_diff(pricing_data_contents, 118, avg_price_1)

    for x in similar_1:
        print pricing_data_contents.iloc[x][['brand_name', 'item_description']]

    pricing_data_contents[pricing_data_contents['brand_name']
                          == "Victoria's Secret"]
    pricing_data_contents.groupby('brand_name').count()

# train = pd.read_csv(
#     '/Users/hslord/kaggle/mercari_price_suggestion/data/train_sample.tsv', delimiter='\t')
# train = train.iloc[:30000]
# train.to_csv(
#     '/Users/hslord/kaggle/mercari_price_suggestion/data/train_sample.csv')
