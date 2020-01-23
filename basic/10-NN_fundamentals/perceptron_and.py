#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:49:32 2020

@author: j-bd
"""

import logging

import numpy as np

from perceptron import Perceptron

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


'''Evaluating the Perceptron Bitwise Datasets with the AND dataset'''

# construct the AND dataset
XS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
YS = np.array([[0], [0], [0], [1]])

# define our perceptron and train it
logging.info(" Training perceptron...")
P = Perceptron(XS.shape[1], alpha=0.1)
P.fit(XS, YS, epochs=20)

# now that our perceptron is trained we can evaluate it
logging.info(" Testing perceptron...")

# now that our network is trained, loop over the data points
for (x, target) in zip(XS, YS):
    # make a prediction on the data point and display the result
    # to our console
    pred = P.predict(x)
    logging.info(f" data={x}, ground-truth={target[0]}, pred={pred}")
