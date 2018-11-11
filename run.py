import numpy as np
import pickle
from sklearn import svm
from sklearn.model_selection import cross_val_score


def build_vector(fn, embeddings):
    # load vocab
    with open('output/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

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
    X = np.zeros((len(lines), embeddings.shape[1]))
    for i, line in enumerate(lines):
        c = 0
        for t in line.strip().split():
            if t in vocab:
                X[i] += embeddings[vocab[t]]
                c += 1
        if c > 0:
            X[i] /= c
    return X


def main():
    # let's load the embeddings
    embeddings = np.load('output/embeddings.npy')

    # now we must build the features.
    train_pos = build_vector('data/train_pos.txt', embeddings)
    train_neg = build_vector('data/train_neg.txt', embeddings)

    X_train = np.r_[train_pos, train_neg]
    y_train = np.r_[np.ones(train_pos.shape[0]), -1 * np.ones(train_neg.shape[0])]

    # let's create a SVM with fixed hyperparameters (we must tune that later on)
    clf = svm.SVC(kernel='linear', C=1, gamma=10**-3)
    scores = cross_val_score(clf, X_train, y_train, cv=3, n_jobs=-1, verbose=True)

    print("Cross validated score: {:.1f} +/- {:.1f}".format(scores.mean() * 100, scores.std() * 100))

    """
    Predictions
    """
    # Set to true if you want to test the model and submit predictions to kaggle
    if True:
        with open('data/test_data.txt') as f:
            id, lines = zip(*map(lambda line: line.split(','), f.readlines()))
        X_test = build_vector(lines, embeddings)
        prediction = clf.predict(X_test)

        np.savetxt("prediction.csv.gz", np.c_[id, prediction], header="Id,Prediction", comments='', delimiter=",",
                   fmt="%d")


if __name__ == '__main__':
    main()
