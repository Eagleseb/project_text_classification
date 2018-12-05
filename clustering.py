import numpy as np
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.cluster import DBSCAN

from loader import load_data, prepare_data


np.random.seed(42)
print("Loading data")
X_train, y_train = load_data('data/glove.twitter.27B/glove.twitter.27B.25d.txt', 'data/train_pos.txt', 'data/train_neg.txt')

print("Preparing data")
X_train, y_train = prepare_data(X_train, y_train)

clustering = DBSCAN(eps=10, min_samples=1000)
y_predict = clustering.fit_predict(X_train)

print(len(np.unique(y_predict)))
