import numpy as np
import pickle
from sklearn import svm, ensemble
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils import shuffle
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

def load_data(glove_fn, train_pos_fn, train_neg_fn, test_fn, scaler=None):
    # let's load the embeddings and the vocab
    print("Loading GloVe")
    # embeddings, vocab = load_glove('output/embeddings.npy', 'output/vocab.pkl')
    embeddings, vocab = load_glove(glove_fn)

    # now we must build the features.
    print("Building vectors")
    train_pos = build_vector(train_pos_fn, embeddings, vocab)
    train_neg = build_vector(train_neg_fn, embeddings, vocab)

    print("Preparing data")
    X_train = np.r_[train_pos, train_neg]
    y_train = np.r_[np.ones(train_pos.shape[0]), -1 * np.ones(train_neg.shape[0])]

    if scaler is None:
        scaler = StandardScaler()
    elif scaler == "no_scale":
        scaler = StandardScaler(with_mean=False, with_std=False)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_train = np.c_[np.ones(len(X_train)), X_train]

    print("Loading test")
    with open(test_fn) as f:
        id, lines = zip(*map(lambda line: line.split(','), f.readlines()))
    X_test = build_vector(lines, embeddings, vocab)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, id
