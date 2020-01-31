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
    '''Get MNIST data and return images with their labels'''
    print("[INFO]: Accessing MNIST...")
    dataset = datasets.fetch_openml("mnist_784")
    input_data = dataset.data

    # If we are using "channels first" ordering, then reshape the design matrix
    # such that the matrix is: num_samples x depth x rows x columns
    if K.image_data_format() == "channels_first":
        input_data = input_data.reshape(input_data.shape[0], 1, 28, 28)
    # Otherwise, we are using "channels last" ordering, so the design matrix
    # shape should be: num_samples x rows x columns x depth
    else:
        input_data = input_data.reshape(input_data.shape[0], 28, 28, 1)

    return input_data, dataset.target.astype("int")

def data_preparation(dataset, labels):
    '''Transform a numpy array (7000, 784) into train / test data and operate
    a scaling plus labelisation'''
    print("[INFO] Scale and split data...")
    # Scale the input data to the range [0, 1]
    input_data = dataset / 255.0
    # Perform a train/test split
    (train_x, test_x, train_y, test_y) = train_test_split(
        input_data, labels, test_size=0.25, random_state=42
    )
    # Convert the labels from integers to vectors
    le = LabelBinarizer()
    train_y = le.fit_transform(train_y)
    test_y = le.transform(test_y)

    return train_x, test_x, train_y, test_y

def main():
    '''Launch main process'''
    input_data, labels = data_loader()

    train_x, test_x, train_y, test_y = data_preparation(input_data, labels)


if __name__ == "__main__":
    main()
