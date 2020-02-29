#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:34:46 2020

@author: j-bd
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_tools import ImageToArrayPreprocessor, SimpleDatasetLoader
from aspectawarepreprocessor import AspectAwarePreprocessor
from minivggnet_tf import MiniVGGNet


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="MiniVGGNet applied to Flowers17 dataset",
        usage='''%(prog)s [with data augmentation]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 minivggnet_flowers17_data_aug.py
        --dataset "path/to/dataset/directory" --output "path/to/model/directory"
        --tboutput "path/to/directory"

        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="path to input dataset"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="directory to save model and plot"
    )
    parser.add_argument(
        "-tb", "--tboutput", required=True, help="path to TensorBoard directory"
    )
    args = vars(parser.parse_args())
    return args

def data_loader(data_directory):
    '''Load data and corresponding labels from disk'''
    print("[INFO] Loading images...")
    # Grab the list of images that we’ll be describing, then extractthe class
    # label names from the image paths
    image_paths = list(paths.list_images(data_directory))
    labels = [species.split(os.path.sep)[-2] for species in image_paths]
    cl_labels = [str(x) for x in np.unique(labels)]

    return image_paths, cl_labels

def data_preparation(image_paths):
    '''Scaling and binarize data'''

    # Initialize the image preprocessors
    aap = AspectAwarePreprocessor(64, 64)
    iap = ImageToArrayPreprocessor()

    # Load the dataset from disk then scale the raw pixel intensities to the
    # range [0, 1]
    sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
    (data, labels) = sdl.load(image_paths, verbose=80)
    data = data.astype("float") / 255.0

    # Partition the data into training and testing splits using 75% of the data
    # for training and the remaining 25% for testing
    (train_x, test_x, train_y, test_y) = train_test_split(
        data, labels, test_size=0.25, random_state=42
    )

    # Convert the labels from integers to vectors
    train_y = LabelBinarizer().fit_transform(train_y)
    test_y = LabelBinarizer().fit_transform(test_y)

    # construct the image generator for data augmentation
    img_augm_gene = ImageDataGenerator(
        rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
        fill_mode="nearest"
    )

    return train_x, test_x, train_y, test_y, img_augm_gene

def checkpoint_call(directory):
    '''Return a callback checkpoint configuration to save only the best model'''
    fname = os.path.sep.join([directory, "weights.hdf5"])
    checkpoint = ModelCheckpoint(
        fname, monitor="val_loss", mode="min", save_best_only=True,
        verbose=1
    )
    return checkpoint

def cnn_training(args, train_x, test_x, train_y, test_y, cl_labels, img_augm_gene):
    '''Launch the MiniVGGNet training'''
    print("[INFO] Compiling model...")
    opt = SGD(lr=0.05)
    model = MiniVGGNet.build(
        width=64, height=64, depth=3, classes=len(cl_labels)
    )
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    model.summary()

    # Callbacks creation
    checkpoint_save = checkpoint_call(args["output"])
    tensor_board = TensorBoard(
        log_dir=args["tboutput"], histogram_freq=1, write_graph=True,
        write_images=True
    )
    callbacks = [checkpoint_save, tensor_board]

    print("[INFO] Training network...")
    history = model.fit_generator(
        img_augm_gene.flow(train_x, train_y, batch_size=32),
        validation_data=(test_x, test_y), steps_per_epoch=len(train_x) // 32,
        epochs=100, callbacks=callbacks, verbose=1
    )
    return history, model

def model_evaluation(model, test_x, test_y, label_names, directory):
    '''Display on terminal command the quality of model's predictions'''
    print("[INFO] Evaluating network...")
    predictions = model.predict(test_x, batch_size=64)

    report = classification_report(
        test_y.argmax(axis=1), predictions.argmax(axis=1),
        target_names=label_names, output_dict=True
    )
    dataframe = pd.DataFrame.from_dict(report, orient="index", )
    dataframe.to_csv(
        os.path.sep.join([directory, "classification_report.csv"]), index=True
    )

    print(
        classification_report(
            test_y.argmax(axis=1), predictions.argmax(axis=1),
            target_names=label_names
        )
    )

def display_learning_evol(history_dic, directory):
    '''Plot the training loss and accuracy'''
    fname = os.path.sep.join([directory, "loss_accuracy_history.png"])
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
    plt.savefig(fname)


def main():
    '''Launch main steps'''
    args = arguments_parser()

    image_paths, cl_labels = data_loader(args["dataset"])

    train_x, test_x, train_y, test_y, img_augm_gene = data_preparation(
        image_paths
    )

    history, model = cnn_training(
        args, train_x, test_x, train_y, test_y, cl_labels, img_augm_gene
    )

    model_evaluation(model, test_x, test_y, cl_labels, args["output"])

    display_learning_evol(history, args["output"])


if __name__ == "__main__":
    main()
