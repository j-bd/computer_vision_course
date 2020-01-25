#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:04:10 2020

@author: j-bd
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets

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
    # define the 784-256-128-10 architecture using Keras
    model = Sequential()
    model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    # we use a softmax activation to obtain normalized class probabilities for
    # each prediction
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

    # grab the MNIST dataset (if this is your first time running this script, the
    # download may take a minute -- the 55MB MNIST dataset will be downloaded)
    logging.info(" Loading MNIST (full) dataset...")
#    dataset = datasets.fetch_mldata("MNIST Original")
    dataset = datasets.fetch_openml("mnist_784")
    # scale the raw pixel intensities to the range [0, 1.0], then
    # construct the training and testing splits
    data = dataset.data.astype("float") / 255.0

    logging.info(" Split dataset...")
    (trainX, testX, trainY, testY) = train_test_split(
        data, dataset.target, test_size=0.25)

    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    model = nn_model()

    # train the model using SGD
    logging.info(" Training network...")
    H = model.fit(
        trainX, trainY, validation_data=(testX, testY), epochs=100,
        batch_size=128
    )

    # evaluate the network
    logging.info(" Evaluating network...")
    predictions = model.predict(testX, batch_size=128)
    logging.info(
        classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
            target_names=[str(x) for x in lb.classes_]
        )
    )

    display_learning_evol(H, args["output"])

if __name__ == "__main__":
    main()
