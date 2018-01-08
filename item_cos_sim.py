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
        - a sparse matrix of cosine similarities between
          reduced item descriptions
    '''
    # Process Data
    MercariFeatureEngineering(
        filename, ',').engineer_features(csv_out_path)
    pricing_contents = pd.read_csv(csv_out_path)
    # Tfidf on lemmatized item descriptions
    vectorizer = TfidfVectorizer().fit(pricing_contents['lemmed_tokens'])
    lem_vectorized = vectorizer.transform(pricing_contents['lemmed_tokens'])
    print lem_vectorized.shape
    print type(pricing_contents.iloc[0]['lemmed_tokens'])
    # Tfidf on brand names
    vectorizer = TfidfVectorizer().fit(pricing_contents['brand_name'])
    brand_vectorized = vectorizer.transform(pricing_contents['brand_name'])
    print brand_vectorized.shape
    # Put into one matrix
    pricing_as_mat = sparse.hstack(
        (lem_vectorized, brand_vectorized))

    # Put into similarity matrix
    pricing_as_mat = cosine_similarity(pricing_as_mat)
    print 'Cosine Similarity Done!'
    return pricing_contents, pricing_as_mat


def x_most_similar(mat, item_idx, x=100):
    '''
    INPUT:
        - mat: cosine similarity matrix for items
        - item_idx: id for row calculating similarities
        - x: number of rows to compare that id to
    OUTPUT:
        - list of indices of x most similar rows
    '''
    top_idxs = np.argpartition(mat, -x)[item_idx, -x:]
    top_idxs = [idx for idx in top_idxs if idx != item_idx]
    return top_idxs


def avg_similar_items(price_mat, similar_idxs):
    """
    INPUT: price_mat: pandas dataframe
           similar_idxs: list of indices
    OUTPUT: Average price of similar items
    """
    avg_price = np.mean(price_mat.iloc[similar_idxs]['price'])
    return avg_price


def price_diff(price_mat, idx, avg_price):
    return price_mat.iloc[idx]['price'] - avg_price


def avg_for_all(mat, price_mat, x=100):
    '''
    INPUT:
        - mat: cosine similarity matrix for items
        - price_mat: pandas dataframe
        - x: number of rows to compare that id to
    OUTPUT:
        - predicted average price for all items
    '''
    predicted_prices = []
    for idx in xrange(mat.shape[0]):
        similar_idxs = x_most_similar(mat, idx, x)
        avg_price = avg_similar_items(price_mat, similar_idxs)
        predicted_prices.append(avg_price)
    return predicted_prices


def rmse_results(predicted_prices, price_mat):
    '''
    INPUT:
        - predicted_prices: list of predicted prices for each item
        - price_mat: pandas dataframe
    OUTPUT:
        - RMSE for model
    '''
    actual_prices = price_mat['price'].values
    rmese = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    return rmse


if __name__ == "__main__":
    pricing_data_contents, pricing_mat = process_pricing_data(
        '/Users/hslord/kaggle/mercari_price_suggestion/data/train_sample.csv',
        'data/new_features_added.csv')
    pricing_data_contents.iloc[990]
    pricing_data_contents.head()
    similar = x_most_similar(pricing_mat, 990)
    avg_price = avg_similar_items(pricing_data_contents, similar)
    price_diff = price_diff(pricing_data_contents, 990, avg_price)
    for x in similar:
        print pricing_data_contents.iloc[x][['brand_name', 'item_description']]


# train = pd.read_csv(
#     '/Users/hslord/kaggle/mercari_price_suggestion/data/train_sample.tsv',
#     delimiter='\t')
# train = train.iloc[:30000]
# train.to_csv(
#     '/Users/hslord/kaggle/mercari_price_suggestion/data/train_sample.csv')
