#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:09:02 2020

@author: j-bd
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras.optimizers import SGD
from keras import backend as K

from lenet import LeNet


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="LeNet CNNs Application",
        usage='''%(prog)s [on MNIST dataset]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 lenet_mnist.py --model path/to/folder/weights.hdf5
        --output path/to/folder/file.png
        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-m", "--model", required=True, help="path to save model"
    )
    parser.add_argument("-o", "--output", required=True, help="path to output")
    args = vars(parser.parse_args())
    return args

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
    label_bin = LabelBinarizer()
    train_y = label_bin.fit_transform(train_y)
    test_y = label_bin.transform(test_y)

    return train_x, test_x, train_y, test_y, label_bin

def training_lenet(train_x, test_x, train_y, test_y, saving_path):
    '''Launch the lenet training'''
    # Initialize the optimizer and model
    print("[INFO] Compiling model...")
    opt = SGD(lr=0.01)
    model = LeNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    model.summary()
    # Train the network
    print("[INFO] Training network...")
    history = model.fit(
        train_x, train_y, validation_data=(test_x, test_y), batch_size=128,
        epochs=20, verbose=1
    )
    model.save(saving_path)

    return history, model

def model_evaluation(model, test_x, test_y, label_bin):
    '''Display on terminal command the quality of model's predictions'''
    print("[INFO] Evaluating network...")
    predictions = model.predict(test_x, batch_size=128)
    print(
        classification_report(
            test_y.argmax(axis=1), predictions.argmax(axis=1),
            target_names=[str(x) for x in label_bin.classes_]
        )
    )

def display_learning_evol(history_dic, saving_path):
    '''Plot the training loss and accuracy'''
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(
        np.arange(0, len(history_dic.history["loss"])),
        history_dic.history["loss"], label="train_loss"
    )
    plt.plot(
        np.arange(0, len(history_dic.history["val_loss"])),
        history_dic.history["val_loss"], label="val_loss"
    )
    plt.plot(
        np.arange(0, len(history_dic.history["accuracy"])),
        history_dic.history["accuracy"], label="train_acc"
    )
    plt.plot(
        np.arange(0, len(history_dic.history["val_accuracy"])),
        history_dic.history["val_accuracy"], label="val_accuracy"
    )
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(saving_path)

def main():
    '''Launch main process'''
    args = arguments_parser()

    input_data, labels = data_loader()

    train_x, test_x, train_y, test_y, label_bin = data_preparation(
        input_data, labels
    )

    history, model = training_lenet(
        train_x, test_x, train_y, test_y, args["model"]
    )

    display_learning_evol(history, args["output"])

    model_evaluation(model, test_x, test_y, label_bin)


if __name__ == "__main__":
    main()
