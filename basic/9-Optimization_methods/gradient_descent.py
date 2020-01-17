#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:40:14 2020

@author: j-bd
"""

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# “blobs" create normally distributed data points – this is a handy function
# when testing or implementing our own models from scratch
from sklearn.datasets import make_blobs

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def arguments_parser():
    '''Get the informations from the operator'''
    parser = argparse.ArgumentParser(
        prog="Gradient descent",
        usage='''%(prog)s [for steps understanding]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 main.py --epochs float --alpha float

        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-e", "--epochs", type=float, default=100, help="# of epochs that "\
        "we’ll use when training our classifier using gradient descent."
    )
    parser.add_argument(
        "-a", "--alpha", type=float, default=0.01, help="learning rate for the"\
        " gradient descent. We typically see 0.1, 0.01, and 0.001 as initial"\
        " learning rate values, but again, this is a hyperparameter you’ll need"\
        " to tune for your own classification problems"
    )
    args = vars(parser.parse_args())
    return args

def sigmoid_activation(x):
    '''Compute the sigmoid activation value for a given input
    When plotted this function will resemble an “S”-shaped curve'''
    return 1.0 / (1 + np.exp(-x))

def predict(X, W):
    '''Make prediction'''
    # take the dot product between our features and weight matrix
    preds = sigmoid_activation(X.dot(W))

    # apply a step function to threshold the outputs to binary
    # class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    # return the predictions
    return preds

def data_generator():
    '''Product a set of train and test data'''
    # generate a 2-class classification problem with 1,000 data points,
    # where each data point is a 2D feature vector
    (X, y) = make_blobs(
        n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1
    )
    y = y.reshape((y.shape[0], 1))

    # insert a column of 1’s as the last entry in the feature
    # matrix -- this little trick allows us to treat the bias
    # as a trainable parameter within the weight matrix
    X = np.c_[X, np.ones((X.shape[0]))]

    # partition the data into training and testing splits using 50% of
    # the data for training and the remaining 50% for testing
    (trainX, testX, trainY, testY) = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    return (trainX, testX, trainY, testY)

def main():
    '''Launch the mains steps'''
    args = arguments_parser()
    logging.info("1. Download data; 2. Split dataset ...")
    (trainX, testX, trainY, testY) = data_generator()

    logging.info("3. Training ...")
    # initialize our weight matrix and list of losses
    W = np.random.randn(trainX.shape[1], 1)
    # keep track of our losses after each epoch
    losses = list()
    # loop over the desired number of epochs
    for epoch in np.arange(0, args["epochs"]):
        # take the dot product between our features ‘X‘ and the weight
        # matrix ‘W‘, then pass this value through our sigmoid activation
        # function, thereby giving us our predictions on the dataset
        preds = sigmoid_activation(trainX.dot(W))

        # now that we have our predictions, we need to determine the
        # ‘error‘, which is the difference between our predictions and
        # the true values
        error = preds - trainY
        # simple loss typically used for binary classification problems
        loss = np.sum(error ** 2)
        losses.append(loss)

        # the gradient descent update is the dot product between our
        # features and the error of the predictions
        # derivee ? pente ?
        gradient = trainX.T.dot(error)

        # in the update stage, all we need to do is "nudge" the weight
        # matrix in the negative direction of the gradient (hence the
        # term "gradient descent" by taking a small step towards a set
        # of "more optimal" parameters
        W += -args["alpha"] * gradient

        # check to see if an update should be displayed
        if epoch == 0 or (epoch + 1) % 5 == 0:
            logging.info(" epoch={}, loss={:.7f}".format(int(epoch + 1), loss))


if __name__ == "__main__":
    main()
