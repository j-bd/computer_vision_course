#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 10:57:17 2020

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
    parser.add_argument(
        "-b", "--batch-size", type=int, default=32,
        help="size of SGD mini-batches"
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

def next_batch(X, y, batchSize):
    '''Create batch from input'''
    # loop over our dataset ‘X‘ in mini-batches, yielding a tuple of
    # the current batched data and labels
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i:i + batchSize], y[i:i + batchSize])

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

def plot(x_val, y_val, args, losses):
    '''Display data'''
    # plot the (testing) classification data
    plt.style.use("ggplot")
    plt.figure()
    plt.title("Data")
    plt.scatter(
        x_val[:, 0], x_val[:, 1], c=np.random.rand(len(y_val)), marker="o", s=30
    )

    # construct a figure that plots the loss over time
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, args["epochs"]), losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()

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
        # initialize the total loss for the epoch
        epoch_loss = list()

        # loop over our data in batches
        for (batchX, batchY) in next_batch(trainX, trainY, args["batch_size"]):
            # take the dot product between our current batch of features
            # and the weight matrix, then pass this value through our
            # activation function
            preds = sigmoid_activation(batchX.dot(W))

            # now that we have our predictions, we need to determine the
            # ‘error‘, which is the difference between our predictions
            # and the true values
            error = preds - batchY
            # simple loss typically used for binary classification problems
            epoch_loss.append(np.sum(error ** 2))

            # the gradient descent update is the dot product between our
            # current batch and the error on the batch
            # derivee ? pente ?
            gradient = batchX.T.dot(error)

            # in the update stage, all we need to do is "nudge" the weight
            # matrix in the negative direction of the gradient (hence the
            # term "gradient descent" by taking a small step towards a set
            # of "more optimal" parameters
            # the weight update stage takes place inside the batch loop – this
            # implies there are multiple weight updates per epoch.
            W += -args["alpha"] * gradient

        # update our loss history by taking the average loss across all
        # batches
        loss = np.average(epoch_loss)
        losses.append(loss)

        # check to see if an update should be displayed
        if epoch == 0 or (epoch + 1) % 5 == 0:
            logging.info(" epoch={}, loss={:.7f}".format(int(epoch + 1), loss))


    # evaluate our model
    logging.info("4. Evaluating ...")
    preds = predict(testX, W)
    print(classification_report(testY, preds))

    plot(testX, testY, args, losses)


if __name__ == "__main__":
    main()
