import pandas as pd
import numpy as np

# Data files
SRC_ARTICLES = '../data/guardian-all/articles-standardized.csv'
SRC_AUTHORS = '../data/guardian-all/authors-standardized.csv'
SRC_COMMENTS = '../data/guardian-all/sorted_comments-standardized.csv'

# If you need to print the full numpy array
# np.set_printoptions(threshold=np.inf)

def get_politic_articles(articles):
    return articles[articles['article_url'].str.contains('/politics/')]

# Most pol articles for 2015 & 2016
def get_years_politic_articles(articles, year):
    return articles[articles['article_url'].str.contains('/politics/{}'.format(year))]

def get_comments_article_ids(comments):
    return comments['article_id'].unique()

def read_chunkwise(comments_filepath, articles, quiet=False):
    article_ids = set(articles['article_id'])
    data_comments_pol_root = []  # Only parent comments
    headline = pd.read_csv(comments_filepath, nrows=10)
    for df_chunk in pd.read_csv(comments_filepath, header=None, skiprows=0, chunksize=1000000):
        if not quiet:
            print('.', end='')
        matches = df_chunk[df_chunk[0].isin(article_ids)]
        # Remove replies
        matches = matches[matches[5].isnull()]  # index 5 is 'parent_comment_id'
        if len(matches):
            if not quiet:
                print('({})'.format(len(matches)), end='')
            data_comments_pol_root.append(matches)
    data_comments_pol_root = pd.concat(data_comments_pol_root)
    data_comments_pol_root.columns = headline.columns  # shape = (40974, 7)
    # Remove parent_comment_id because it's NaN everywhere in this dataframe
    columns = list(headline.columns)
    columns.remove('parent_comment_id')
    data_comments_pol_root = data_comments_pol_root[columns]  # shape = (40974, 6)
    return data_comments_pol_root

def read_all_chunkwise(comments_filepath, articles, quiet=False):
    article_ids = set(articles['article_id'])
    data_comments_pol_root = []  # Only parent comments
    data_comments_pol_all = []  # Includes replies
    headline = pd.read_csv(comments_filepath, nrows=10)
    for df_chunk in pd.read_csv(comments_filepath, header=None, skiprows=0, chunksize=1000000):
        if not quiet:
            print('.', end='')
        matches = df_chunk[df_chunk[0].isin(article_ids)]
        if len(matches):
            print(len(matches), end='')
            data_comments_pol_all.append(matches)

            # Remove replies
            matches = matches[matches[5].isnull()]  # index 5 is 'parent_comment_id'
            if len(matches):
                if not quiet:
                    print('({})'.format(len(matches)), end='')
                data_comments_pol_root.append(matches)
    data_comments_pol_root = pd.concat(data_comments_pol_root)
    data_comments_pol_root.columns = headline.columns  # shape = (40974, 7)
    # Remove parent_comment_id because it's NaN everywhere in this dataframe
    columns = list(headline.columns)
    columns.remove('parent_comment_id')
    data_comments_pol_root = data_comments_pol_root[columns]  # shape = (40974, 6)

    data_comments_pol_all = pd.concat(data_comments_pol_all)
    data_comments_pol_all.columns = headline.columns  # shape = (?, 7)
    return data_comments_pol_root, data_comments_pol_all

def load_data(comments_filepath):
    data_articles = pd.read_csv(SRC_ARTICLES)  # shape = (626395, 2)
    data_authors = pd.read_csv(SRC_AUTHORS)
    data_comments_pol = pd.read_csv(comments_filepath)
    return (data_articles, data_authors, data_comments_pol)

# Add rel_upvotes, total_upvotes and rank to comments df
def normalize(comments):
    print('Add rank attribute')
    comments['rank'] = comments.groupby('article_id')['timestamp'].rank(method='dense').astype(int)

    print('Add article attributes')
    comments['total_upvotes'] = comments.groupby('article_id')['upvotes'].transform('sum')
    comments['total_comments'] = comments.groupby('article_id')['upvotes'].transform('count')

    print('Apply normalization')
    comments['rel_upvotes'] = comments.apply(lambda row: (row.upvotes / (row.total_upvotes or 1)) * 100, axis=1)
    return co

def extract_features(comments):
    print('Preparing features')
    upvotes_per_author = comments[['author_id', 'upvotes']].groupby('author_id').sum().iloc[:, 0]
    comments_per_author = comments[['author_id', 'upvotes']].groupby('author_id').count().iloc[:, 0]
    mean_upvotes_per_author = upvotes_per_author / comments_per_author

    replies_count = pd.Series(index=comments['comment_id'], data=0)
    parent_comment_ids = comments['parent_comment_id']
    for parent_id in parent_comment_ids[~parent_comment_ids.isnull()].values:
        if parent_id in replies_count:
            replies_count.loc[parent_id] += 1

    print('Collecting features')
    extracted_features = dict()
    extracted_features['upvotes'] = comments['upvotes'].values
    extracted_features['replies'] = replies_count.loc[comments['comment_id']].values
    extracted_features['text_length'] = comments['comment_text'].str.len().values
    extracted_features['author_upvotes'] = upvotes_per_author.loc[comments['author_id']].values
    extracted_features['author_comments'] = comments_per_author.loc[comments['author_id']].values
    extracted_features['mean_author_upvotes'] = mean_upvotes_per_author.loc[comments['author_id']].values

    extracted_features['rank'] = comments.groupby('article_id')['timestamp'].rank(method='dense').astype(int)
    extracted_features['total_upvotes'] = comments.groupby('article_id')['upvotes'].transform('sum')
    extracted_features['total_comments'] = comments.groupby('article_id')['upvotes'].transform('count')
    extracted_features['rel_upvotes'] = [
        (upvotes / (total_upvotes or 1)) * 100
        for upvotes, total_upvotes
        in zip(extracted_features['upvotes'], extracted_features['total_upvotes'])
    ]
    # extracted_features['rel_upvotes'] = extracted_features.apply(lambda row: (row.upvotes / (row.total_upvotes or 1)) * 100, axis=1)

    return pd.DataFrame.from_dict(extracted_features)
