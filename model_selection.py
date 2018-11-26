import numpy as np
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from loader import load_data, prepare_data


def main():
    np.random.seed(42)
    print("Loading data")
    X_train, y_train = load_data('data/glove.twitter.27B.25d.txt', 'data/train_pos.txt', 'data/train_neg.txt')

    print("Preparing data")
    X_train, y_train = prepare_data(X_train, y_train)

    # let's create a SVM with fixed hyperparameters (we must tune that later on)
    # clf = svm.SVC(kernel='linear', C=10)
    # -> SVM are a bad choice because we have too much data
    clf = ensemble.RandomForestClassifier(n_estimators=100)
    param_grid = [
        # {'n_estimators': [10, 20, 50, 100]},
        {'max_depth': np.int(np.sqrt(X_train.shape[1])) * np.array([1, 2, 10, 100])}
    ]

    cv = GridSearchCV(clf, param_grid=param_grid, cv=3, return_train_score=True)

    print("Cross validating")
    cv.fit(X_train, y_train)
    print(cv.cv_results_)
    # print("Cross validated score: {:.1f} +/- {:.1f}".format(scores.mean() * 100, scores.std() * 100))


if __name__ == '__main__':
    main()
