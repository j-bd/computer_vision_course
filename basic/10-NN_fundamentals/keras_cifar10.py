#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 21:15:24 2020

@author: j-bd
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def arguments_parser():
    '''Get the informations from the operator'''
    parser = argparse.ArgumentParser(
        prog="MNIST",
        usage='''%(prog)s [Neural Network Application]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 keras_mnist.py --output path/to/folder/file.png

        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="path to the output loss/accuracy plot"
    )
    args = vars(parser.parse_args())
    return args

def nn_model():
    '''Define the structure of the neural network'''
    # define the 3072-1024-512-10 architecture using Keras
    model = Sequential()
    model.add(Dense(1024, input_shape=(3072,), activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    sgd = SGD(0.01)
    model.compile(
        loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
    )
    return model

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
    '''Launch the mains steps'''
    args = arguments_parser()

    # load the training and testing data, scale it into the range [0, 1],
    # then reshape the design matrix
    logging.info(" Loading CIFAR-10 data...")
    ((trainX, trainY), (testX, testY)) = cifar10.load_data()
    trainX = trainX.astype("float") / 255.0
    testX = testX.astype("float") / 255.0
    trainX = trainX.reshape((trainX.shape[0], 3072))
    testX = testX.reshape((testX.shape[0], 3072))

    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    # initialize the label names for the CIFAR-10 dataset
    labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
    "horse", "ship", "truck"]

    model = nn_model()

    logging.info(" Training network...")
    H = model.fit(
        trainX, trainY, validation_data=(testX, testY), epochs=100,
        batch_size=32
    )

    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=labelNames))

    display_learning_evol(H, args["output"])

if __name__ == "__main__":
    main()
