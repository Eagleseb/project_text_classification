import numpy as np
from keras.models import load_model

from loader import load_data_nn

reload_model = False
training = True
max_words, emb_dim = 64, 200

glove_fn = 'data/glove.twitter.27B/glove.twitter.27B.{}d.txt'.format(emb_dim)
train_pos_fn, train_neg_fn = 'data/train_pos_full.txt', 'data/train_neg_full.txt'


def main():
    np.random.seed(42)
    print("Loading data")
    embeddings, vocab, x_train, y_train, x_test, test_id = load_data_nn(glove_fn, train_pos_fn, train_neg_fn,
                                                                        'data/test_data.txt', max_words=max_words)

    print("Building model")
    model = load_model('output/rnn_model.epoch3.h5')

    print("Predicting")
    prediction = model.predict_classes(x_test, batch_size=128, verbose=1)
    prediction[prediction == 0] = -1
    np.savetxt("output/prediction.csv", np.c_[test_id, prediction],
               header="Id,Prediction", comments='', delimiter=",", fmt="%d")


if __name__ == '__main__':
    main()
