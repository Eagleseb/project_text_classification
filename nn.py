"""
Keras nn implementation following
https://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html
instructions
"""

import numpy as np
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

from loader import load_data, prepare_data

input_dim = 200

np.random.seed(42)
print("Loading data")
data = load_data('data/glove.twitter.27B/glove.twitter.27B.{}d.txt'.format(input_dim),
                 'data/train_pos.txt', 'data/train_neg.txt', p=0)

print("Preparing data")
X, y = prepare_data(*data)

x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.2)

print("Building model")
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=input_dim))
model.add(Dense(32, activation='relu', input_dim=64))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Training")
model.fit(x_train, y_train, epochs=9, batch_size=32, verbose=2)

print("Testing")
score = model.evaluate(x_test, y_test, batch_size=128, verbose=2)
print(score[1])
