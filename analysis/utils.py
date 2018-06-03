import datetime
import os
import re
import urllib
import sys
import math
import numpy as np
import pandas as pd
from scipy.stats import describe

# Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

from wordcloud import WordCloud
import networkx as nx
from gensim.models import KeyedVectors

# Data files
SRC_ARTICLES = '../data/guardian-all/articles-standardized.csv'
SRC_AUTHORS = '../data/guardian-all/authors-standardized.csv'
SRC_COMMENTS = '../data/guardian-all/sorted_comments-standardized.csv'
# SRC_COMMENTS_POL = '../data/guardian-all/sorted_comments-standardized-pol.csv'
SRC_COMMENTS_POL_ALL = '../data/guardian-all/sorted_comments-standardized-pol-all.csv'
# SRC_COMMENTS_POL_TEXT = '../data/guardian-all/sorted_comments-standardized-pol-text.csv'
SRC_COMMENTS_TOKENIZED_BIN = '../data/guardian-all/genzim-guardian-comments-50-tokenized.bin'


def plot_history(history):
    plt.plot(history.history['acc'], 'b', label='Train Acc')
    plt.plot(history.history['loss'], 'r', label='Train Loss')
    plt.plot(history.history['val_acc'], 'b--', label='Val Acc')
    plt.plot(history.history['val_loss'], 'r--', label='Val Loss')
    plt.legend()

def plot_timestamps(article_id):
    datetimes = data[data['article_id'] == article_id]['timestamp']
    timestamps = [int(datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").timestamp()) for x in datetimes]
    plt.hist(timestamps)

def load_embedding():
    print('Loading embeddings...')
    word_vectors = KeyedVectors.load(SRC_COMMENTS_TOKENIZED_BIN)
    return word_vectors

def load_data():
    data_articles = pd.read_csv(SRC_ARTICLES)  # shape = (626395, 2)
    data_articles_pol = data_articles[data_articles['article_url'].str.contains('/politics/')]  # shape = (20167, 2)
    data_authors = pd.read_csv(SRC_AUTHORS)
    data_comments_pol = pd.read_csv(SRC_COMMENTS_POL_ALL)
    return (data_articles, data_articles_pol, data_authors, data_comments_pol)

def article_ids(comments):
    return comments['article_id'].unique()  # 2876 ids

def extract_features():
    print('Preparing features')
    upvotes_per_author = data_comments_pol[['author_id', 'upvotes']].groupby('author_id').sum().iloc[:, 0]
    comments_per_author = data_comments_pol[['author_id', 'upvotes']].groupby('author_id').count().iloc[:, 0]
    mean_upvotes_per_author = upvotes_per_author / comments_per_author

    replies_count = pd.Series(index=data_comments_pol['comment_id'], data=0)
    parent_comment_ids = data_comments_pol['parent_comment_id']
    for parent_id in parent_comment_ids[~parent_comment_ids.isnull()].values:
        if parent_id in replies_count:
            replies_count.loc[parent_id] += 1
    
    print('Collecting features')
    extracted_features = dict()
    extracted_features['upvotes'] = data_comments_pol['upvotes'].values
    extracted_features['replies'] = replies_count.loc[data_comments_pol['comment_id']].values
    extracted_features['text_length'] = data_comments_pol['comment_text'].str.len().values
    extracted_features['author_upvotes'] = upvotes_per_author.loc[data_comments_pol['author_id']].values
    extracted_features['author_comments'] = comments_per_author.loc[data_comments_pol['author_id']].values
    extracted_features['mean_author_upvotes'] = mean_upvotes_per_author.loc[data_comments_pol['author_id']].values

    return pd.DataFrame.from_dict(extracted_features)

