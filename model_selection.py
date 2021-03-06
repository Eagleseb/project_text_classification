import numpy as np
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression

from loader import load_data, prepare_data


def main():
    np.random.seed(42)
    print("Loading data")
    data = load_data('data/glove.twitter.27B/glove.twitter.27B.200d.txt', 'data/train_pos.txt',
                     'data/train_neg.txt', p=10**-4.4)

    print("Preparing data")
    X_train, y_train = prepare_data(*data)

    # let's create a SVM with fixed hyperparameters (we must tune that later on)
    # clf = svm.SVC(kernel='linear', C=10)
    # -> SVM are a bad choice because we have too much data
    # clf = LogisticRegression()
    clf = ensemble.RandomForestClassifier(n_estimators=100)

    print("Cross validating")
    scores = cross_val_score(clf, X_train, y_train, cv=3)
    print("Cross validated score: {:.1f} +/- {:.1f}".format(scores.mean() * 100, scores.std() * 100))


if __name__ == '__main__':
    main()
