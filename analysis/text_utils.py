import re
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText

SRC_COMMENTS_TOKENIZED_BIN = '../data/embedding/gensim-guardian-comments-50-tokenized.bin'

def load_embedding():
    print('Loading embeddings...')
    model = FastText.load_fasttext_format(SRC_COMMENTS_TOKENIZED_ORIG_BIN)
    word_vectors = model.wv  # Map from word to word vector
    del model # Save memory
    # word_vectors.vocab is sorted by occurrences (e.g. see vocab['.'].count)
    # weights_matrix = word_vectors.syn0  # Matrix with 50D vector for ~1.3 mio words
    # word_vectors['woman'] == weights_matrix[word2index['woman']]
    return word_vectors

def pad_or_cut_tokenized_comments(X_train, X_test, word_index, max_sequence_length):
    train_sequences = [[word_index.get(t, 0) for t in comment]
                       for comment in X_train]
    test_sequences = [[word_index.get(t, 0)  for t in comment]
                      for comment in X_test]
    train_data = pad_sequences(train_sequences, maxlen=max_sequence_length,
                               padding="pre", truncating="post")
    print('Shape of training data tensor:', train_data.shape)
    test_data = pad_sequences(test_sequences, maxlen=max_sequence_length,
                              padding="pre", truncating="post")
    print('Shape of test_data tensor:', test_data.shape)
    return train_data, test_data

def inspect_preprocessed_comment(comments, X_train, backtranslater, idx):
    print('Original:')
    print(comments.iloc[idx])
    print('\nAfter preprocessing (& backtranslating):')
    print(' '.join([backtranslater[x] for x in X_train[idx] if x in backtranslater]))
