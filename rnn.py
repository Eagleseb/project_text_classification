"""
Keras rnn implementation following
https://heartbeat.fritz.ai/coreml-with-glove-word-embedding-and-recursive-neural-network-part-2-d72c1a66b028
"""

import numpy as np
from keras.layers import Embedding, Dense, GRU, LSTM, \
    TimeDistributed, Activation, Flatten, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import TensorBoard

from loader import load_data_nn

reload_model = False
training = False
max_words, emb_dim = 64, 200

glove_fn = 'data/glove.twitter.27B/glove.twitter.27B.{}d.txt'.format(emb_dim)
train_pos_fn, train_neg_fn = 'data/train_pos_full.txt', 'data/train_neg_full.txt'

np.random.seed(42)
print("Loading data")
if training:
    embeddings, vocab, X, y = load_data_nn(glove_fn, train_pos_fn, train_neg_fn, max_words=max_words)

    ixes = np.random.permutation(X.shape[0])
    train, test = ixes[:len(ixes) * 8 // 10], ixes[len(ixes) * 8 // 10:]
    x_train, x_test, y_train, y_test = X[train], X[test], y[train], y[test]
else:
    embeddings, vocab, x_train, y_train, x_test, test_id = load_data_nn(glove_fn, train_pos_fn, train_neg_fn,
                                                                        'data/test_data.txt', max_words=max_words)

print("Building model")
if reload_model:
    model = load_model('output/rnn_model.epoch3.h5')
else:
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
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
print("Training")
if training:
    history = model.fit(x_train, y_train, callbacks=[tensorBoardCallback], validation_data=(x_test, y_test), epochs=1, batch_size=128, verbose=1)
else:
    history = model.fit(x_train, y_train, callbacks=[tensorBoardCallback], epochs=1, batch_size=128, verbose=1)
# model.save('output/rnn_model.h5')

print("Testing")
if training:
    score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print(score[1])
else:
    prediction = model.predict_classes(x_test, batch_size=128, verbose=1)
    prediction[prediction == 0] = -1
    np.savetxt("output/prediction.csv", np.c_[test_id, prediction],
               header="Id,Prediction", comments='', delimiter=",", fmt="%d")
