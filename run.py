import numpy as np
from sklearn import ensemble

from loader import load_data, prepare_data


def main():
    np.random.seed(42)
    print("Loading data")
    X_train, y_train, X_test, test_id = load_data('data/glove.twitter.27B/glove.twitter.27B.25d.txt',
                                             'data/train_pos.txt', 'data/train_neg.txt', 'data/test_data.txt')

    print("Preparing data")
    X_train, y_train, X_test, test_id = prepare_data(X_train, y_train, X_test, test_id)

    # let's create a SVM with fixed hyperparameters (we must tune that later on)
    # clf = svm.SVC(kernel='linear', C=10)
    # -> SVM are a bad choice because we have too much data
    clf = ensemble.RandomForestClassifier(n_estimators=100)
    # clf = LogisticRegression(C=1)
    # clf = RidgeClassifier()

    print("Training")
    clf.fit(X_train, y_train)
    print("Predicting")
    prediction = clf.predict(X_test)

    np.savetxt("prediction.csv.gz", np.c_[test_id, prediction], header="Id,Prediction", comments='', delimiter=",",
               fmt="%d")


if __name__ == '__main__':
    main()
