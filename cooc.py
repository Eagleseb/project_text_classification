#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle


def main():
    with open('output/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    data, row, col = [], [], []
    counter = 1
    for fn in ['data/train_pos_full.txt', 'data/train_neg_full.txt']:
        with open(fn) as f:
            for line in f:
                tokens = [vocab[t] for t in line.strip().split() if t in vocab]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1
    cooc = coo_matrix((data, (row, col)))
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    with open('output/cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
