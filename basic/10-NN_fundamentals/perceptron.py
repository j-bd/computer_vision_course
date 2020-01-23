#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 21:47:35 2020

@author: j-bd
"""

import numpy as np


class Perceptron:
    '''Implementation of a perceptron'''
    def __init__(self, n, alpha=0.1):
        '''Initialize the weight matrix and store the learning rate with 'n'
        the number of explicatives variables and alpha the learning rate'''
        # files our weight matrix W with random values sampled from a “normal”
        # (Gaussian) distribution with zero mean and unit variance
        # The weight matrix will have N + 1 entries, one for each of the N
        # inputs in the feature vector, plus one for the bias
        # We divide W by the square-root of the number of inputs, a common
        # technique used to scale our weight matrix, leading to faster
        # convergence
        self.weight = np.random.randn(n + 1) / np.sqrt(n)
        self.alpha = alpha

    def step(self, x):
        '''Apply the step function (activation function)'''
        return 1 if x > 0 else 0

    def fit(self, xs, y, epochs=10):
        '''Fit the model to the data'''
        # 'x' as input, 'y' as target output class labels
        # insert a column of 1’s as the last entry in the feature
        # matrix -- this little trick allows us to treat the bias
        # as a trainable parameter within the weight matrix
        xs = np.c_[xs, np.ones((xs.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point
            for (x, target) in zip(xs, y):
                # take the dot product between the input features
                # and the weight matrix, then pass this value
                # through the step function to obtain the prediction
                p = self.step(np.dot(x, self.weight))

                # only perform a weight update if our prediction
                # does not match the target
                if p != target:
                    # determine the error
                    error = p - target

                    # update the weight matrix
                    self.weight += -self.alpha * error * x

    def predict(self, xs, addBias=True):
        '''Predict the class labels for a given set of input data'''
        # ensure our input is a matrix
        xs = np.atleast_2d(xs)

        # check to see if the bias column should be added
        if addBias:
            # insert a column of 1’s as the last entry in the feature
            # matrix (bias)
            xs = np.c_[xs, np.ones((xs.shape[0]))]

        # take the dot product between the input features and the
        # weight matrix, then pass the value through the step
        # function
        return self.step(np.dot(xs, self.weight))
