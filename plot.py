import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from loader import load_data


def pca(X_train, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    distrib = np.r_[0, np.cumsum(pca.explained_variance_ratio_)]

    x = np.arange(0, n_components + 1)
    plt.bar(x, distrib)
    plt.axhline(y=.95, xmin=0, xmax=n_components, label="95% threshold", c="r")
    plt.axvline(x=(n_components+1)**.95 - 1, ymin=0, ymax=1, c="r")
    plt.plot(x, np.log(x + 1) / np.log(n_components + 1), label="log(x + 1)/log(n_components + 1)", c="black")
    plt.legend()
    plt.title("Cumulative explained variance of\nthe {} principal components out of {} features"
              .format(n_components, X_train.shape[1]))
    plt.savefig("output/explained_variance.eps")


def main():
    print("Loading data")
    X_train, y_train = load_data('data/glove.twitter.27B.200d.txt', 'data/train_pos.txt', 'data/train_neg.txt')
    print("Plotting explained variance")
    pca(X_train, X_train.shape[1])


if __name__ == '__main__':
    main()
