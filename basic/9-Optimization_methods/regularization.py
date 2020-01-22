#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:50:59 2020

@author: j-bd
"""
import argparse
import logging

from imutils import paths
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import data_tools

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def arguments_parser():
    '''Get the informations from the operator'''
    parser= argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", required=True, help="path to input dataset"
    )
    args = vars(parser.parse_args())
    return args

def main():
    '''Launch the mains steps'''
    args = arguments_parser()

    logging.info(" Step 1: loading images...")
    image_paths = list(paths.list_images(args["dataset"]))

    logging.info(" Step 1: preprocessing images ...")
    sp = data_tools.SimplePreprocessor(32, 32)
    sdl = data_tools.SimpleDatasetLoader(preprocessors=[sp])
    (data, labels) = sdl.load(image_paths, verbose=500)
    data = data.reshape((data.shape[0], 3072))

    logging.info(" Step 2: Split data...")
    # encode the labels as integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(
        data, labels, test_size=0.25, random_state=42
    )

    logging.info(" Step 3: training...")
    # loop over our set of regularizers
    for r in (None, "l1", "l2"):
        # train a SGD classifier using a softmax loss function and the
        # specified regularization function for 10 epochs
        logging.info(f" Training model with {r} penalty")
        # We’ll be using cross-entropy loss, with regularization penalty of r
        # and a default λ of 0.0001. We’ll use SGD to train the model for 10
        # epochs with a learning rate of α = 0.01
        model = SGDClassifier(
            loss="log", penalty=r, max_iter=10, learning_rate="constant",
            eta0=0.01, random_state=42
        )
        model.fit(trainX, trainY)

        # evaluate the classifier
        acc = model.score(testX, testY)
        logging.info(" ‘{}‘ penalty accuracy: {:.2f}%".format(r, acc * 100))


if __name__ == "__main__":
    main()
