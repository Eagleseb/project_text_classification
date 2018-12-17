"""
Keras nn implementation following
https://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html
instructions
"""

import numpy as np
from keras.layers import Embedding, Dense, GRU, LSTM, \
    TimeDistributed, Activation, Flatten
from keras.models import Sequential
from keras.callbacks import TensorBoard

from loader import load_data_nn

max_words, emb_dim = 100, 200

np.random.seed(42)
print("Loading data")
embeddings, vocab, X, y = load_data_nn('data/glove.twitter.27B/glove.twitter.27B.{}d.txt'.format(emb_dim),
                                       'data/train_pos_full.txt', 'data/train_neg_full.txt', max_words=max_words)

ixes = np.random.permutation(X.shape[0])
train, test = ixes[:len(ixes) * 8 // 10], ixes[len(ixes) * 8 // 10:]
x_train, x_test, y_train, y_test = X[train], X[test], y[train], y[test]

print("Building model")
model = Sequential([
    Embedding(len(vocab), emb_dim, weights=[embeddings], input_length=max_words, trainable=False),
    GRU(emb_dim, batch_size=1, input_shape=(None, emb_dim), return_sequences=True),
    # LSTM(emb_dim, batch_size=1, input_shape=(None, emb_dim), return_sequences=True),
    TimeDistributed(Dense(64)),
    Activation('relu'),
    TimeDistributed(Dense(32)),
    Activation('relu'),
    Flatten(),
    Dense(1, activation='sigmoid')
])
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training")
model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=2)

print("Testing")
score = model.evaluate(x_test, y_test, batch_size=128, verbose=2)
print(score[1])
