import numpy as np
import pickle
from sklearn import svm, ensemble
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score


def load_glove(fn, vocab=None):
    # Load our own vocab and embeddings
    if vocab:
        with open(vocab, 'rb') as f:
            vocab = pickle.load(f)
        embeddings = np.load(fn)
    # Or load some embeddings we downloaded here: https://nlp.stanford.edu/projects/glove/
    else:
        with open(fn) as f:
            lines = f.readlines()
            vocab, embeddings = zip(*map(lambda r: (r[0], np.array(r[1:], dtype=np.float)), map(str.split, lines)))
            vocab = {t: i for i, t in enumerate(vocab)}
    return np.asarray(embeddings), vocab


def build_vector(fn, embeddings, vocab):
    # fn can be either a filename or the tweets directly
    if isinstance(fn, str):
        with open(fn) as f:
            lines = f.readlines()
    else:
        lines = fn

    # for each tweet
    #   for each vocab word in that tweet
    #       vec += embedding[word]
    #   vec = mean(vec)
    X = np.zeros((len(lines), len(embeddings[0])))
    for i, line in enumerate(lines):
        c = 0
        for t in line.strip().split():
            if t in vocab:
                X[i] += embeddings[vocab[t]]
                c += 1
        if c > 0:
            X[i] /= c
    return X

def load_data(glove_fn, train_pos_fn, train_neg_fn, test_fn=None):
    """
    Load the dataset
    :param glove_fn:
    :param train_pos_fn:
    :param train_neg_fn:
    :param test_fn: (optionnal) if provided, return the test vectors along their ids aswell
    :return: X_train, y_train or X_train, y_train, X_test, test_id if test_fn is provided
    """
    # let's load the embeddings and the vocab
    # embeddings, vocab = load_glove('output/embeddings.npy', 'output/vocab.pkl')
    embeddings, vocab = load_glove(glove_fn)

    vocab = remove_stopwords(train_pos_fn, train_neg_fn, vocab, p=0)

    # now we must build the features.
    train_pos = build_vector(train_pos_fn, embeddings, vocab)
    train_neg = build_vector(train_neg_fn, embeddings, vocab)

    X_train = np.r_[train_pos, train_neg]
    y_train = np.r_[np.ones(train_pos.shape[0]), -1 * np.ones(train_neg.shape[0])]

    if test_fn is not None:
        with open(test_fn) as f:
            test_id, lines = zip(*map(lambda line: line.split(','), f.readlines()))
        X_test = build_vector(lines, embeddings, vocab)
        return X_train, y_train, X_test, test_id
    else:
        return X_train, y_train


def prepare_data(X_train, y_train, X_test=None, test_id=None, scaler=None, random_state=42):
    """
    Perform feature engineering on the dataset
    :param X_train:
    :param y_train:
    :param X_test:
    :return:
    """
    if scaler is None:
        scaler = StandardScaler()
    elif scaler == "no_scale":
        scaler = StandardScaler(with_mean=False, with_std=False)

    # Let's create a pipeline to transform the data
    # according to the pca, (n_features+1)**p - 1 account for p*100% of the dataset total variance
    # it's a lower bound
    pipeline = Pipeline([('pca', PCA(np.int(np.ceil((X_train.shape[1]+1)**.95 - 1)))), ('scaler', scaler)])

    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
    X_train = pipeline.fit_transform(X_train)

    if X_test is not None:
        X_test, test_id = shuffle(X_test, test_id, random_state=random_state)
        X_test = pipeline.transform(X_test)

        return X_train, y_train, X_test, test_id
    else:
        return X_train, y_train


def get_frequencies(fn):
    # fn can be either a filename or the tweets directly
    if isinstance(fn, str):
        with open(fn) as f:
            lines = f.readlines()
    else:
        lines = fn

    n = len(lines)
    frequencies = {}
    for line in lines:
        for t in set(line.strip().split()):
            frequencies[t] = (frequencies.get(t, 0) * n + 1.) / n
    return frequencies


def remove_stopwords(train_pos_fn, train_neg_fn, vocab, p):
    """
    Remove words whose frequencies is the same up to p in positive and negative samples.
    :param freq_pos:
    :param freq_neg:
    :param p:
    :param vocab:
    :return:
    """
    freq_pos = get_frequencies(train_pos_fn)
    freq_neg = get_frequencies(train_neg_fn)
    vocab = vocab.copy()
    for t in list(vocab.keys()):
        if np.abs(freq_pos.get(t, 0) - freq_neg.get(t, 0)) < p:
            del vocab[t]
    return vocab
