#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:09:02 2020

@author: j-bd
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras.optimizers import SGD
from keras import backend as K

from lenet import LeNet


def data_loader():
    '''Get MNIST data and prepare it'''
    print("[INFO]: Accessing MNIST...")
    dataset = datasets.fetch_openml("mnist_784")
    data = dataset.data

    # If we are using "channels first" ordering, then reshape the design matrix
    # such that the matrix is: num_samples x depth x rows x columns
    if K.image_data_format() == "channels_first":
        data = data.reshape(data.shape[0], 1, 28, 28)
    # Otherwise, we are using "channels last" ordering, so the design matrix
    # shape should be: num_samples x rows x columns x depth
    else:
        data = data.reshape(data.shape[0], 28, 28, 1)
    return data

def main():
    '''Launch main process'''
    data = data_loader()


if __name__ == "__main__":
    main()
