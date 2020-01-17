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


def sigmoid_activation(x):
    '''Compute the sigmoid activation value for a given input
    When plotted this function will resemble an “S”-shaped curve'''
    return 1.0 / (1 + np.exp(-x))

