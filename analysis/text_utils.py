import re
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors

SRC_COMMENTS_TOKENIZED_BIN = '../data/guardian-all/genzim-guardian-comments-50-tokenized.bin'

def load_embedding():
    print('Loading embeddings...')
    embedding_model = KeyedVectors.load(SRC_COMMENTS_TOKENIZED_BIN)
    return embedding_model

# Replace urls
re_url = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\
                    .([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
                    re.MULTILINE|re.UNICODE)
# Replace ips
re_ip = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")


def text_to_wordlist(tokenizer, vocab, text, lower=False):
    # Looks like all URLs are removed
    text = re_url.sub("URL", text)
    # But there some IPs we'd like to replace
    text = re_ip.sub("IPADDRESS", text)
    # Tokenize
    text = tokenizer.tokenize(text)
    # optional: lower case
    if lower:
        text = [t.lower() for t in text]
    # Return a list of words
    vocab.update(text)
    # Return a list of words
    return text

def process_comments(tokenizer, vocab, X, lower=False):
    # We have no NaN words but it could be useful
    list_sentences = list(X.fillna("NAN_WORD").values)
    comments = []
    for text in tqdm(list_sentences):
        txt = text_to_wordlist(tokenizer, vocab, text, lower=lower)
        comments.append(txt)
    return comments

def pad_or_cut_tokenized_comments(most_common, comments, train_size, max_nb_words, max_sequence_length):
    # Select most common words
    word_index = {t[0]: i+1 for i,t in enumerate(most_common[:max_nb_words])}
    train_sequences = [[word_index.get(t, 0) for t in comment]
                       for comment in comments[:train_size]]
    test_sequences = [[word_index.get(t, 0)  for t in comment] 
                      for comment in comments[train_size:]]
    
    # Pad sequences
    train_data = pad_sequences(train_sequences, maxlen=max_sequence_length, 
                               padding="pre", truncating="post")
    print('Shape of training data tensor:', train_data.shape)

    test_data = pad_sequences(test_sequences, maxlen=max_sequence_length, padding="pre",
                              truncating="post")
    print('Shape of test_data tensor:', test_data.shape)

    return train_data, test_data, word_index

def get_weights_matrix(word_index, word_vectors, nb_words, wv_dim):
    # we initialize the matrix with random numbers
    wv_matrix = (np.random.rand(nb_words, wv_dim) - 0.5) / 5.0

    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        try:
            # words not found in embedding index will be all-zeros.
            wv_matrix[i] = word_vectors[word]
        except:
            pass
    return wv_matrix