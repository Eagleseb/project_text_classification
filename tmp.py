from keras.layers import LSTM, Convolution1D, Flatten, Dropout, Dense
from keras.models import Sequential

input_shape = (200, 25)
model = Sequential()
model.add(Convolution1D(64, 3, input_shape=input_shape, padding='same'))
model.add(Convolution1D(32, 3, padding='same'))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])