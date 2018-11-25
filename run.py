import numpy as np
import pickle
from sklearn import svm, ensemble
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

from loader import load_data

def main():
    X_train, y_train, X_test, id = load_data('data/glove.twitter.27B.200d.txt',
                                             'data/train_pos.txt', 'data/train_neg.txt', 'data/test_data.txt')

    print("Training")
    # let's create a SVM with fixed hyperparameters (we must tune that later on)
    # clf = svm.SVC(kernel='linear', C=10)
    clf = ensemble.RandomForestClassifier(n_estimators=100)
    scores = cross_val_score(clf, X_train, y_train, cv=3, n_jobs=-1)

    print("Cross validated score: {:.1f} +/- {:.1f}".format(scores.mean() * 100, scores.std() * 100))

    """
    Predictions
    """
    # Set to true if you want to test the model and submit predictions to kaggle
    if False:
        print("Predicting")
        prediction = clf.predict(X_test)

        np.savetxt("prediction.csv.gz", np.c_[id, prediction], header="Id,Prediction", comments='', delimiter=",",
                   fmt="%d")


if __name__ == '__main__':
    main()
