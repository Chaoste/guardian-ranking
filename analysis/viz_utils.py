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


def plot_history(history):
    plt.plot(history.history['acc'], 'b', label='Train Acc')
    plt.plot(history.history['loss'], 'r', label='Train Loss')
    plt.plot(history.history['val_acc'], 'b--', label='Val Acc')
    plt.plot(history.history['val_loss'], 'r--', label='Val Loss')
    plt.legend()

def plot_categories_pie(articles):
    regex = re.compile('https://www\.theguardian\.com/([^/]*)(/(.*)|$)')
    sections = [regex.match(x).group(1) for x in articles['article_url'] if regex.match(x)]
    mismatches = [x for x in articles['article_url'].unique() if not regex.match(x)]
    print(f'Not matching URLs: {"  ".join(mismatches)}')

    section_counts = pd.DataFrame(sections).groupby(0).apply(lambda x: len(x))
    section_counts = list(zip(section_counts.index, section_counts.values))
    section_counts = np.array(sorted(section_counts, reverse=True, key=lambda x: x[1]))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.pie(section_counts[:, 1], labels=section_counts[:, 0], autopct='%1.1f%%',
           explode=[x == 'politics' for x in section_counts[:, 0]],shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    return section_counts

def plot_timestamps(article_id):
    datetimes = data[data['article_id'] == article_id]['timestamp']
    timestamps = [int(datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").timestamp()) for x in datetimes]
    plt.hist(timestamps)

def plot_feature_correlations(extracted_features_df, author_comments=0):
    labels = list(reversed(extracted_features_df.columns))

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    cax = ax.matshow(extracted_features_df[extracted_features_df['author_comments'] > author_comments][labels].corr())
    cbar = fig.colorbar(cax)
    ax.set_xticklabels([''] + labels, rotation=45, ha='left')
    ax.set_yticklabels([''] + labels)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title(rf'Correlations: Comment Features ($comments_{{author}} \geq {author_comments}$)', y=1.15)
    fig.savefig(f'correlations-{author_comments}.pdf')
