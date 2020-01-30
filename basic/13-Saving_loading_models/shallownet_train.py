#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:00:24 2020

@author: j-bd
"""

import argparse

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD

from data_tools import ImageToArrayPreprocessor
from data_tools import SimplePreprocessor
from data_tools import SimpleDatasetLoader
from shallownet import ShallowNet


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Serialization",
        usage='''%(prog)s [Loading and saving]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 shallownet_train.py --dataset path/to/folder/containing_image
        --model path/to/folder/weights.hdf5 --output path/to/folder/file.png

        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="path to input dataset"
    )
    parser.add_argument(
        "-m", "--model", required=True, help="path to output model"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="path to output"
    )
    args = vars(parser.parse_args())
    return args

def preprocessing(arg):
    '''Prepare data and labels under numpy format'''
    # Grab the list of images that we’ll be describing
    image_paths = list(paths.list_images(arg))

    # Initialize the image preprocessors
    resize = SimplePreprocessor(32, 32)
    image_to_array = ImageToArrayPreprocessor()

    # Load the dataset from disk then scale the raw pixel intensities to the
    # range [0, 1]
    dataset_loader = SimpleDatasetLoader(preprocessors=[resize, image_to_array])
    (data, labels) = dataset_loader.load(image_paths, verbose=50)
    data = data.astype("float") / 255.0
    return data, labels

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

    print(" Loading and preprocessing images...")
    data, labels = preprocessing(args["dataset"])

    print(" Splitting data...")
    (train_x, test_x, train_y, test_y) = train_test_split(
        data, labels, test_size=0.25, random_state=42
    )

    # Convert the labels from integers to vectors
    train_y = LabelBinarizer().fit_transform(train_y)
    test_y = LabelBinarizer().fit_transform(test_y)

    # Initialize the optimizer and model
    print(" Compiling model...")
    opt = SGD(lr=0.005)
    model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    model.summary()

    # Train the network
    print(" Training network...")
    history = model.fit(
        train_x, train_y, validation_data=(test_x, test_y), batch_size=32,
        epochs=100, verbose=1
    )

    # Save the network to disk
    print(" Serializing network...")
    model.save(args["model"])

    # evaluate the network
    print(" Evaluating network...")
    predictions = model.predict(test_x, batch_size=32)
    print(
        classification_report(
            test_y.argmax(axis=1), predictions.argmax(axis=1),
            target_names=["fries", "beans", "potatoes"]
        )
    )

    display_learning_evol(history, args["output"])


if __name__ == "__main__":
    main()
