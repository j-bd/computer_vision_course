#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:44:31 2020

@author: j-bd
"""

import os
import argparse

import matplotlib
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from keras.datasets import cifar10

from trainingmonitor import TrainingMonitor
from minivggnet import MiniVGGNet


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Monitoring training",
        usage='''%(prog)s [MiniVGGNet on Cifar10 dataset]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 cifar10_monitor.py --model path/to/folder/weights.hdf5
        --output path/to/folder
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
    '''Get Cifar10 data'''
    print("[INFO] Loading CIFAR-10 data...")
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    return train_x, test_x, train_y, test_y

def data_preparation(train_x, test_x, train_y, test_y):
    '''Scaling and labelising data'''
    train_x = train_x.astype("float") / 255.0
    test_x = test_x.astype("float") / 255.0

    # Convert the labels from integers to vectors
    label_bin = LabelBinarizer()
    train_y = label_bin.fit_transform(train_y)
    test_y = label_bin.transform(test_y)

    # Initialize the label names for the CIFAR-10 dataset
    label_names = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
        "ship", "truck"
    ]

    return train_x, test_x, train_y, test_y, label_names

def training_minivggnet(train_x, test_x, train_y, test_y, saving_path):
    '''Launch the lenet training'''
    # Initialize the optimizer and model
    print("[INFO] Compiling model...")
    opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    model.summary()

    # train the network
    print("[INFO] Training network...")
    history = model.fit(
        train_x, train_y, validation_data=(test_x, test_y), batch_size=64,
        epochs=40, verbose=1
    )
    model.save(saving_path)

    return history, model

def monitoring(folder_path):
    '''Monitor the training'''
    # construct the set of callbacks
    fig_path = os.path.sep.join([folder_path, "{}.png".format(os.getpid())])
    json_path = os.path.sep.join([folder_path, "{}.json".format(os.getpid())])
    callbacks = [TrainingMonitor(fig_path, json_path=json_path)]
    return callbacks

def main():
    '''Launch main steps'''
    args = arguments_parser()

    # Show information on the process ID
    print("[INFO] Process ID: {}".format(os.getpid()))

    train_x, test_x, train_y, test_y = data_loader()

    train_x, test_x, train_y, test_y, label_names = data_preparation(
        train_x, test_x, train_y, test_y
    )

    history, model = training_minivggnet(
        train_x, test_x, train_y, test_y, args["model"]
    )

    monitoring(args["output"])


if __name__ == "__main__":
    main()
