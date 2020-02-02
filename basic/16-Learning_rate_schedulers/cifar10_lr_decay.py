#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 18:28:53 2020

@author: j-bd
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10

from minivggnet import MiniVGGNet


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="MiniVGGNet CNNs Application with Learning Rate Schedulers",
        usage='''%(prog)s [on Cifar10 dataset]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 minivggnet_cifar10.py --model path/to/folder/weights.hdf5
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

def step_decay(epoch):
    '''Initialize the base initial learning rate, drop factor, and epochs to
    drop every'''
    init_alpha = 0.01
    factor = 0.5
    drop_every = 5

    # Compute learning rate for the current epoch
    alpha = init_alpha * (factor ** np.floor((1 + epoch) / drop_every))

    return float(alpha)

def training_minivggnet(train_x, test_x, train_y, test_y, callbacks, saving_path):
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
        epochs=40, callbacks=callbacks, verbose=1
    )
    model.save(saving_path)

    return history, model

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

def model_evaluation(model, test_x, test_y, label_names):
    '''Display on terminal command the quality of model's predictions'''
    print("[INFO] Evaluating network...")
    predictions = model.predict(test_x, batch_size=64)
    print(
        classification_report(
            test_y.argmax(axis=1), predictions.argmax(axis=1),
            target_names=label_names
        )
    )

def main():
    '''Launch main process'''
    args = arguments_parser()

    train_x, test_x, train_y, test_y = data_loader()

    train_x, test_x, train_y, test_y, label_names = data_preparation(
        train_x, test_x, train_y, test_y
    )

    # Define the set of callbacks to be passed to the model during training
    callbacks = [LearningRateScheduler(step_decay)]

    history, model = training_minivggnet(
        train_x, test_x, train_y, test_y, callbacks, args["model"]
    )

    display_learning_evol(history, args["output"])

    model_evaluation(model, test_x, test_y, label_names)

if __name__ == "__main__":
    main()
