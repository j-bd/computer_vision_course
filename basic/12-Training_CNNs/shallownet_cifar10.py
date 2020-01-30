#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:56:56 2020

@author: j-bd
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import cifar10

from shallownet import ShallowNet


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Cifar 10 Classification",
        usage='''%(prog)s [Application of ShallowNet CNNs structure]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 shallownet_cifar10.py --output path/to/folder/file.png

        All arguments are mandatory.
        '''
    )
    parser.add_argument("-o", "--output", required=True, help="path to output")
    args = vars(parser.parse_args())
    return args

def data_loader():
    '''Load the training and testing data, then scale it into the range [0, 1]'''
    print("[INFO] loading CIFAR-10 data...")
    ((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
    train_x = train_x.astype("float") / 255.0
    test_x = test_x.astype("float") / 255.0

    # convert the labels from integers to vectors
    lab_bin = LabelBinarizer()
    train_y = lab_bin.fit_transform(train_y)
    test_y = lab_bin.transform(test_y)
    # initialize the label names for the CIFAR-10 dataset
    label_names = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
        "ship", "truck"
    ]

    return train_x, train_y, test_x, test_y, label_names

def display_learning_evol(fit_dic, save_path):
    '''Plot the training loss and accuracy'''
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(
        np.arange(0, len(fit_dic.history["loss"])), fit_dic.history["loss"],
        label="train_loss"
    )
    plt.plot(
        np.arange(0, len(fit_dic.history["val_loss"])),
        fit_dic.history["val_loss"], label="val_loss"
    )
    plt.plot(
        np.arange(0, len(fit_dic.history["accuracy"])),
        fit_dic.history["accuracy"], label="train_acc"
    )
    plt.plot(
        np.arange(0, len(fit_dic.history["val_accuracy"])),
        fit_dic.history["val_accuracy"], label="val_accuracy"
    )
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(save_path)

def main():
    '''Launch main process'''
    args = arguments_parser()

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
        epochs=4, verbose=1
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

    display_learning_evol(history, args["output"])


if __name__ == "__main__":
    main()
