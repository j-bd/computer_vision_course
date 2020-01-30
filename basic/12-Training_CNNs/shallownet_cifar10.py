#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:56:56 2020

@author: j-bd
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import cifar10

from shallownet import ShallowNet


def data_loader():
    '''Load the training and testing data, then scale it into the range [0, 1]'''
    print("[INFO] loading CIFAR-10 data...")
    ((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
    train_x = train_x.astype("float") / 255.0
    test_x = test_x.astype("float") / 255.0

    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    train_y = lb.fit_transform(train_y)
    test_y = lb.transform(test_y)
    # initialize the label names for the CIFAR-10 dataset
    label_names = ["airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"]

    return train_x, train_y, test_x, test_y, label_names


def main():
    '''Launch main process'''
    train_x, train_y, test_x, test_y, label_names = data_loader()

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.01)
    model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )

    # train the network
    print("[INFO] training network...")
    history = model.fit(
        train_x, train_y, validation_data=(test_x, test_y), batch_size=32,
        epochs=40, verbose=1
    )

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(test_x, batch_size=32)
    print(
        classification_report(
            test_y.argmax(axis=1), predictions.argmax(axis=1),
            target_names=label_names
        )
    )


if __name__ == "__main__":
    main()
