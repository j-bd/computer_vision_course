#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 18:28:53 2020

@author: j-bd
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10

from minivggnet import MiniVGGNet


def step_decay(epoch):
    '''Initialize the base initial learning rate, drop factor, and epochs to
    drop every'''
    init_alpha = 0.01
    factor = 0.25
    drop_every = 5

    # Compute learning rate for the current epoch
    alpha = init_alpha * (factor ** np.floor((1 + epoch) / drop_every))

    return float(alpha)

def main():
    '''Launch main process'''


if __name__ == "__main__":
    main()
