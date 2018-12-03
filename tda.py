import matplotlib.pyplot as plt
import numpy as np
import phstuff.barcode as bc
import phstuff.diphawrapper as dipha
from sklearn.metrics.pairwise import cosine_distances

from loader import load_data

print("Loading data")
X_train, y_train = load_data('data/glove.twitter.27B/glove.twitter.27B.25d.txt',
                             'data/train_pos.txt', 'data/train_neg.txt')

print("Computing cosine kernel")
K = pairwise_kernels(X_train[:300], metric='cosine')
K /= K.max()
K = np.arccos(K)

print("Computing barcode")
runner = dipha.DiphaRunner(2, cpu_count=1, dipha='/Users/sebastien.morand/dev/dipha/dipha')
runner.weight_matrix(K)
runner.run()

for interval in runner.barcode[1]:
    print(interval)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bc.plot(ax, runner.barcode[1], K.min(), K.max())
plt.show()
