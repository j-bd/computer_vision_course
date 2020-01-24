#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:37:36 2020

@author: j-bd
"""

import logging

import numpy as np

from neuralnetwork import NeuralNetwork

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


'''Evaluating the NeuralNetwork with the XOR dataset'''

# construct the XOR dataset
XS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
YS = np.array([[0], [1], [1], [0]])

# define our 2-2-1 neural network and train it
nn = NeuralNetwork([2, 2, 1], alpha=0.5)
logging.info(f" Neural Network shape: {nn}")

logging.info(" Training Neural Network...")
nn.fit(XS, YS, epochs=20000)

# now that our network is trained, loop over the XOR data points
for (x, target) in zip(XS, YS):
    # make a prediction on the data point and display the result
    # to our console
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    logging.info(
        " Data={}, ground-truth={}, pred={:.4f}, step={}".format(
            x, target[0], pred, step
        )
    )
