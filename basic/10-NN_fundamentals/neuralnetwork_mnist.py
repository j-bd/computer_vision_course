#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:01:57 2020

@author: j-bd
"""

import logging

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

from neuralnetwork import NeuralNetwork

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


# load the MNIST dataset and apply min/max scaling to scale the pixel intensity
# values to the range [0, 1] (each image is represented by an 8 x 8 = 64-dim
# feature vector)
logging.info(" Loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
# perform min/max normalizing by scaling each digit into the range [0, 1]
data = (data - data.min()) / (data.max() - data.min())
logging.info(f" Samples: {data.shape[0]}, dim: {data.shape[1]}")

# construct the training and testing splits
logging.info(" Operating of train/test split...")
(trainX, testX, trainY, testY) = train_test_split(
    data, digits.target, test_size=0.25
)
# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# train the network
logging.info(" Training network...")
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
logging.info(f" {nn}")
nn.fit(trainX, trainY, epochs=1000)

# evaluate the network
logging.info(" Evaluating network...")
predictions = nn.predict(testX)
# To find the class label with the largest probability for each data point, we
# use the argmax function. This function will return the index of the label with
# the highest predicted probability
predictions = predictions.argmax(axis=1)
logging.info(classification_report(testY.argmax(axis=1), predictions))
