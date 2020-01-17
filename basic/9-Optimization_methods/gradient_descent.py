#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:40:14 2020

@author: j-bd
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# “blobs" create normally distributed data points – this is a handy function
# when testing or implementing our own models from scratch
from sklearn.datasets import make_blobs


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

def main():
    '''Launch the mains steps'''
    args = arguments_parser()


if __name__ == "__main__":
    main()
