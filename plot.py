import numpy as np
import pickle
from sklearn import svm, ensemble
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy as sp

from loader import load_data

def main():
    X_train, y_train, _, _ = load_data('data/glove.twitter.27B.25d.txt', 'data/train_pos.txt', 'data/train_neg.txt', 'data/test_data.txt')

    eigvals = sp.linalg.eigvals(X_train.dot(X_train.T))
    print(eigvals / np.sum(eigvals))

if __name__ == '__main__':
    main()