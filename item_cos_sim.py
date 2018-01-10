import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from time import time
from feature_engineering import MercariFeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


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
    lem_svd = TruncatedSVD(n_components=250)
    lem_transform = lem_svd.fit_transform(lem_vectorized)
    # Tfidf on brand names
    vectorizer = TfidfVectorizer().fit(pricing_contents['brand_name'])
    brand_vectorized = vectorizer.transform(pricing_contents['brand_name'])
    brand_svd = TruncatedSVD(n_components=100)
    brand_transform = brand_svd.fit_transform(brand_vectorized)
    # Put into one matrix
    # pricing_as_mat = sparse.hstack(
    #     (lem_vectorized, brand_vectorized))
    pricing_as_mat = np.concatenate(
        (lem_transform, brand_transform), axis=1)
    print 'Matrix Returned!'
    # Put into similarity matrix
    pricing_as_mat = cosine_similarity(pricing_as_mat)
    # Making corresponding index similarities = 0 instead of 1
    np.place(pricing_as_mat, pricing_as_mat == 1, [0])
    print 'Cosine Similarity Done!'
    return pricing_contents, pricing_as_mat


def x_most_similar(mat, x=100):
    '''
    INPUT:
        - mat: cosine similarity matrix for items
        - x: number of rows to compare that id to
    OUTPUT:
        - list of indices of x most similar rows
    '''
    top_idxs = np.argpartition(mat, -x, axis=1)[:, -x:]
    return top_idxs


def avg_for_all(top_mat, price_mat):
    '''
    INPUT:
        - top_mat: matrix containing top x most similar indices to each item
        - price_mat: pandas dataframe
        - x: number of rows to compare that id to
    OUTPUT:
        - predicted average price for all items
    '''
    predicted_prices = []
    for idx in xrange(top_mat.shape[0]):
        sim_idx = top_mat[idx]
        avg_price = np.mean(price_mat.iloc[sim_idx]['price'])
        predicted_prices.append(avg_price)
    print 'Prices Predicted!'
    return predicted_prices


def results_metrics(predicted_prices, price_mat):
    '''
    INPUT:
        - predicted_prices: list of predicted prices for each item
        - price_mat: pandas dataframe
    OUTPUT:
        - RMSE for model
    '''
    actual_prices = price_mat['price'].values
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    rmsle = sum((np.log(predicted_prices) - np.log(actual_prices))
                ** 2) / len(predicted_prices)
    print 'Metrics Calculated!'
    return rmse, rmsle


if __name__ == "__main__":
    train = pd.read_csv(
        '/Users/hslord/kaggle/mercari_price_suggestion/data/train.tsv',
        delimiter='\t')
    # train = train.iloc[:10000]
    train.to_csv(
        '/Users/hslord/kaggle/mercari_price_suggestion/data/train.csv')
    pricing_data_contents, cos_mat = process_pricing_data(
        '/Users/hslord/kaggle/mercari_price_suggestion/data/train.csv',
        'data/new_features_added.csv')
    top_similarities = x_most_similar(cos_mat, x=100)
    predicted_prices_sample = avg_for_all(
        top_similarities, pricing_data_contents)
    rmse_sample, rmsle_sample = results_metrics(
        predicted_prices_sample, pricing_data_contents)
    print rmsle_sample
    # for x in similar:
    #     print pricing_data_contents.iloc[x][['brand_name', 'item_description']]
