#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:22:48 2020

@author: j-bd
"""

import logging

import numpy as np

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class NeuralNetwork:
    '''Simulate a Neaural Network'''
    def __init__(self, layers, alpha=0.1):
        '''Initialize the list of weights matrices, then store the network
        architecture and learning rate
        "layers": A list of integers which represents the actual architecture of
        the feedforward network. For example, a value of [2, 2, 1] would imply
        that our first input layer has two nodes, our hidden layer has two
        nodes, and our final output layer has one node
        alpha: The learning rate of our neural network'''
        self.weights = []
        self.layers = layers
        self.alpha = alpha

        # start looping from the index of the first layer but stop before we
        # reach the last two layers
        for i in np.arange(0, len(layers) - 2):
            # randomly initialize a weight matrix connecting the number of nodes
            # in each respective layer together, adding an extra node for the bias
            weight = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            # normalization
            self.weights.append(weight / np.sqrt(layers[i]))

        # the last two layers are a special case where the input connections
        # need a bias term but the output does not
        weight = np.random.randn(layers[-2] + 1, layers[-1])
        self.weights.append(weight / np.sqrt(layers[-2]))

    def __repr__(self):
        '''This function is useful for debugging'''
        # construct and return a string that represents the network architecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        '''Activation function'''
        # compute and return the sigmoid activation value for a given input value
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        '''Derivative of the sigmoid function'''
        # compute the derivative of the sigmoid function ASSUMING that ‘x‘ has
        # already been passed through the ‘sigmoid‘ function
        return x * (1 - x)

    def fit_partial(self, x, y):
        '''Heart of the backpropagation algorithm'''
        # construct our list of output activations for each layer as our data
        # point flows through the network; the first activation is a special
        # case -- it’s just the input feature vector itself
        A = [np.atleast_2d(x)]

        # FEEDFORWARD:
        # loop over the layers in the network
        for layer in np.arange(0, len(self.weights)):
            # feedforward the activation at the current layer by taking the dot
            # product between the activation and the weight matrix -- this is
            # called the "net input" to the current layer
            net = A[layer].dot(self.weights[layer])

            # computing the "net output" is simply applying our nonlinear
            # activation function to the net input
            out = self.sigmoid(net)

            # once we have the net output, add it to our list of activations
            A.append(out)

        # BACKPROPAGATION
        # the first phase of backpropagation is to compute the difference between
        # our *prediction* (the final output activation in the activations list)
        # and the true target value
        error = A[-1] - y

        # from here, we need to apply the chain rule and build our list of
        # deltas ‘D‘; the first entry in the deltas is simply the error of the
        # output layer times the derivative of our activation function for the
        # output value
        D = [error * self.sigmoid_deriv(A[-1])]

        # once you understand the chain rule it becomes super easy to implement
        # with a ‘for‘ loop -- simply loop over the layers in reverse order
        # (ignoring the last two since we already have taken them into account)
        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the delta of the
            # *previous layer* dotted with the weight matrix of the current
            # layer, followed by multiplying the delta by the derivative of the
            # nonlinear activation function for the activations of the current
            # layer
            delta = D[-1].dot(self.weights[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # since we looped over our layers in reverse order we need to reverse
        # the deltas
        D = D[::-1]

        # WEIGHT UPDATE PHASE
        # loop over the layers
        for layer in np.arange(0, len(self.weights)):
            # update our weights by taking the dot product of the layer
            # activations with their respective deltas, then multiplying this
            # value by some small learning rate and adding to our weight matrix
            # -- this is where the actual "learning" takes place
            self.weights[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def calculate_loss(self, input_data, targets):
        '''Make predictions for the input data points then compute the loss'''
        targets = np.atleast_2d(targets)
        predictions = self.predict(input_data, add_bias=False)
        # Compute the sum squared error
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        # return the loss
        return loss

    def fit(self, XS, YS, epochs=1000, display_update=100):
        '''Train the NN'''
        # insert a column of 1’s as the last entry in the feature matrix -- this
        # little trick allows us to treat the bias as a trainable parameter
        # within the weight matrix
        XS = np.c_[XS, np.ones((XS.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point and train our network on it
            for (x, target) in zip(XS, YS):
                self.fit_partial(x, target)

            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % display_update == 0:
                loss = self.calculate_loss(XS, YS)
                logging.info(" Epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def predict(self, input_data, add_bias=True):
        '''Predict an output regarding input data'''
        # initialize the output prediction as the input features -- this value
        # will be (forward) propagated through the network to obtain the final
        # prediction
        data = np.atleast_2d(input_data)

        # check to see if the bias column should be added
        if add_bias:
            # insert a column of 1’s as the last entry in the feature matrix
            # (bias)
            data = np.c_[data, np.ones((data.shape[0]))]

        # loop over our layers in the network
        for layer in np.arange(0, len(self.weights)):
            # computing the output prediction is as simple as taking the dot
            # product between the current activation value ‘p‘ and the weight
            # matrix associated with the current layer, then passing this value
            # through a nonlinear activation function
            data = self.sigmoid(np.dot(data, self.weights[layer]))

        # return the predicted value
        return data
