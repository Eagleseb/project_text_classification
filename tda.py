import matplotlib.pyplot as plt
import numpy as np
import phstuff.barcode as bc
import phstuff.diphawrapper as dipha
from sklearn.metrics import pairwise_kernels
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.decomposition import PCA

from loader import load_data

import sys
sys.path.append('/home/sebastien/dev/DREiMac')
from CircularCoordinates import CircularCoords

print("Loading data")
X_train, y_train = load_data('data/glove.twitter.27B/glove.twitter.27B.25d.txt',
                             'data/train_pos.txt', 'data/train_neg.txt')

# print("Computing cosine kernel")
# K = pairwise_kernels(resample(X_train[y_train == -1], n_samples=100), metric='cosine')
# K /= K.max()
# K = np.arccos(K)

# print("Computing barcode")
# runner = dipha.DiphaRunner(2, cpu_count=1, dipha='./output/dipha')
# runner.weight_matrix(K)
# runner.run()
#
# for interval in runner.barcode[1]:
#     print(interval)
#
# x = list(map(lambda b: b.birth, runner.barcode[1]))
# y = list(map(lambda b: b.death - b.birth, runner.barcode[1]))
#
# plt.scatter(x, y)
# plt.show()

sss = StratifiedShuffleSplit(train_size=1000)
ix, _ = list(sss.split(X_train, y_train))[0]
X, y = X_train[ix], y_train[ix]

for prime in [5]:
    res = CircularCoords(X, 100, prime=prime, cocycle_idx=[0])
    plt.clf()
    plt.scatter(np.cos(res['thetas']), np.sin(res['thetas']), 10, y, edgecolor='none')
    plt.axis('equal')
    plt.colorbar()
    plt.title("Prime {}".format(prime))
    plt.show()
    res.pop("rips")
