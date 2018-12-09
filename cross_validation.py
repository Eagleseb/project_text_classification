#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:28:22 2018

@author: Darcane
"""

import numpy as np
from sklearn import ensemble

from loader import load_data, prepare_data
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

from feature_expansion import feature_expansion


np.random.seed(42)
print("Loading data")
X_train, y_train, X_test, test_id = load_data('data/glove.twitter.27B/glove.twitter.27B.25d.txt',
                                         'data/train_pos.txt', 'data/train_neg.txt', 'data/test_data.txt')

improved_X_train = feature_expansion(X_train)

print("Preparing data")
improved_X_train, y_train = prepare_data(improved_X_train, y_train)

# let's create a SVM with fixed hyperparameters (we must tune that later on)
# clf = svm.SVC(kernel='linear', C=10)
# -> SVM are a bad choice because we have too much data
clf = ensemble.RandomForestClassifier(n_estimators=10)
# clf = LogisticRegression(C=1)
# clf = RidgeClassifier()


print("Training")
clf.fit(improved_X_train, y_train)
scores = cross_val_score(clf, improved_X_train, y_train, cv=3, n_jobs=-1)

print("Cross validated score: {:.1f} +/- {:.1f}".format(scores.mean() * 100, scores.std() * 100))
         
