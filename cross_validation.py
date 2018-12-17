#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:28:22 2018

@author: Darcane
"""

import numpy as np
from sklearn import ensemble
import matplotlib.pyplot as plt

from loader import load_data, prepare_data
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_validate

from feature_expansion import feature_expansion


np.random.seed(42)
# improved_X_train = feature_expansion(X_train)

# let's create a SVM with fixed hyperparameters (we must tune that later on)
# clf = svm.SVC(kernel='linear', C=10)
# -> SVM are a bad choice because we have too much data
clf = ensemble.RandomForestClassifier(n_estimators=10)
# clf = LogisticRegression(C=1)
# clf = RidgeClassifier()

test_scores = []

ps = np.logspace(-5, -4, 5)
for p in ps:
    print("Loading data")
    data = load_data('data/glove.twitter.27B/glove.twitter.27B.25d.txt',
                             'data/train_pos_full.txt', 'data/train_neg_full.txt', p=p)
    print("Preparing data")
    X_train, y_train = prepare_data(*data)
    print("Training")
    scores = cross_val_score(clf, X_train, y_train, cv=3, n_jobs=-1)
    test_scores.append(scores)

    # print("Cross validated score: {:.1f} +/- {:.1f}".format(
    #    scores['test_score'].mean() * 100, scores['test_score'].std() * 100))

test_scores = np.array(test_scores)

mean_test = np.mean(test_scores, 1)
std_test = np.std(test_scores, 1)

plt.errorbar(ps, mean_test, std_test, c="b", label="Test score")
plt.semilogx()
plt.savefig('output/remove_lowfreq.eps')

best = np.argmax(mean_test)
print("Best accuracy: {:.2f} with p=10**{:.2f}".format(mean_test[best], np.log10(ps[best])))
