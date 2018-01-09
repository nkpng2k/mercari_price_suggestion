import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
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
    pricing_df = pd.read_csv(csv_out_path)
    # Tfidf on lemmatized item descriptions
    vectorizer = TfidfVectorizer().fit(pricing_df['lemmed_tokens'])
    lem_vectorized = vectorizer.transform(pricing_df['lemmed_tokens'])
    # Tfidf on brand names
    vectorizer = TfidfVectorizer().fit(pricing_df['brand_name'])
    brand_vectorized = vectorizer.transform(pricing_df['brand_name'])
    # Put into one matrix
    pricing_as_mat = sparse.hstack(
        (lem_vectorized, brand_vectorized))
    return pricing_df, pricing_as_mat


def cos_sim(train_mat, test_mat):
    '''
    INPUT:
        - train_mat: a tfidf vectorized train matrix
        - test_mat: a tfidf vectorized test matrix
    OUTPUT:
        - a similarity matrix for each test row compared to the train matrix
    '''
    test_similarity = cosine_similarity(test_mat, train_mat)
    return test_similarity


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
    # get desired number of train data points
    train = pd.read_csv(
        '/Users/hslord/kaggle/mercari_price_suggestion/data/train.tsv',
        delimiter='\t')
    train = train.iloc[:5000]
    train.to_csv(
        '/Users/hslord/kaggle/mercari_price_suggestion/data/train_sample.csv')
    # get desired number of test data points
    test = pd.read_csv(
        '/Users/hslord/kaggle/mercari_price_suggestion/data/test.tsv',
        delimiter='\t')
    test = test.iloc[:5000]
    test.to_csv(
        '/Users/hslord/kaggle/mercari_price_suggestion/data/test_sample.csv')
    # pre-process train data
    train_df, train_sample_mat = process_pricing_data(
        '/Users/hslord/kaggle/mercari_price_suggestion/data/train_sample.csv',
        'data/new_features_added.csv')
    # pre-process test data
    test_df, test_sample_mat = process_pricing_data(
        '/Users/hslord/kaggle/mercari_price_suggestion/data/test_sample.csv',
        'data/new_features_added.csv')
    # cosine similarity of test compared to train
    cos_sim_test = cos_sim(train_sample_mat, test_sample_mat)
    # get x most similar
    top_similarities = x_most_similar(cos_mat, x=100)
    # predict prices
    predicted_prices_sample = avg_for_all(
        top_similarities, pricing_data_contents)
    # get metrics
    rmse_sample, rmsle_sample = results_metrics(
        predicted_prices_sample, pricing_data_contents)
