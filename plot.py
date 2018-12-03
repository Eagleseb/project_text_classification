import matplotlib.pyplot as plt
import numpy as np
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score


from loader import load_data, prepare_data



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
    
    
def n_estimators_computation(num_estimators, X_train, y_train):

    clf = ensemble.RandomForestClassifier(n_estimators=num_estimators)

    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train, cv=3, n_jobs=-1)
    
    scores_mean = scores.mean() * 100
    print("Cross validated score: {:.1f} +/- {:.1f}".format(scores.mean() * 100, scores.std() * 100))
             
    return scores_mean
    
def n_estimators_plot(lower_b, upper_b, num_interval):
    
    values = np.logspace(np.log10(lower_b), np.log10(upper_b), num = num_interval)
    scores_estimators = np.zeros(len(values))
    
    
    np.random.seed(42)
    
    print("Loading data")
    X_train, y_train = load_data('data/glove.twitter.27B/glove.twitter.27B.25d.txt',
                                             'data/train_pos.txt', 'data/train_neg.txt')

    print("Preparing data")
    X_train, y_train = prepare_data(X_train, y_train)
    
    
    for idx, v in enumerate(values) :
        print("Computation for the number of estimators : {}".format(int(np.floor(v))))
        scores_estimators[idx] = n_estimators_computation(int(np.floor(v)), X_train, y_train)
    
    plt.figure()
    plt.plot(values, scores_estimators, label = "Scores for different values of number of estimators")
    # plt.axhline(y=.95, xmin=0, xmax=n_components, label="95% threshold", c="r")
    # plt.axvline(x=(n_components+1)**.95 - 1, ymin=0, ymax=1, c="r")
    plt.legend()
    plt.title("Performance on the cross-validation for different numbers of estimators with Random Forest Classifier")
        
    return scores_estimators

def main():
#    print("Loading data")
#    X_train, y_train = load_data('data/glove.twitter.27B/glove.twitter.27B.200d.txt',
#                                 'data/train_pos.txt', 'data/train_neg.txt')
#    print("Plotting explained variance")
#    pca(X_train, X_train.shape[1])

    print("Plotting the performance depending on the number of estimators")
    n_estimators_plot(1, 50, 10)
    
    


if __name__ == '__main__':
    main()
