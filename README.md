# Project Text Sentiment Classification

## Abstract:

The aim of this project is to classify tweets into a 'happy' and a 'sad' class.
To fulfill that aim, we implemented different machine learning algorithms and
finally choose a Recurrent Neural Network as the best method.
We achieved an accuracy of about 86% which is close to the state of the art.

## Requirements:
- Python 3
- Tensorflow
- Keras
- ScikitLearn

## How to reproduce:

1. Download the datasets on [CrowdAI](https://www.crowdai.org/challenges/epfl-ml-text-classification/dataset_files)
2. Put them into a directory called data/
3. Download twitter GloVe embeddings over [here](http://nlp.stanford.edu/data/glove.twitter.27B.zip)
4. Put them into a directory called data/glove.twitter.27B
5. Create an empty directory called output/ and put into it the trained model you downloaded over [here](https://we.tl/t-tR2JoaqPrR)
6. Run the file run.py with python 3. It should produce a file prediction.csv that you can upload directly on CrowdAI

## Content:
- **run.py**, a simple script that load the saved model and predict the classes of the test dataset
- **mode_selection.py**, a simple script that we used to cross-validate the accuracy of different classification algorithms
- **rnn.py**, the implementation of our Recurrent Neural Network
- **plot.py**, different functions that we used to make the plot of the report