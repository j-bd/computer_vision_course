#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:08:27 2020

@author: j-bd
"""

import os
import argparse

import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import utils
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from imutils import paths

from lenet_tf import LeNet


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Case study",
        usage='''%(prog)s [Training LeNet CNNs]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 train_model.py
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
    data = []
    labels = []

    for image_path in sorted(list(paths.list_images(data_directory))):
        # Load the image, pre-process it, and store it in the data list
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = imutils.resize(image, width=28)
        image = img_to_array(image)
        data.append(image)
        # Extract the class label from the image path and update the labels list
        label = image_path.split(os.path.sep)[-3]
        label = "smiling" if label == "positives" else "not_smiling"
        labels.append(label)
    return data, labels

def data_preparation(dataset, labels):
    '''Scaling and binarize data'''
    dataset = np.array(dataset, dtype="float") / 255.0
    labels = np.array(labels)

    # Convert the labels from integers to vectors
    label_enc = LabelEncoder()
    labels = utils.to_categorical(label_enc.fit_transform(labels), 2)

    # Account for skew in the labeled data -> unbalanced data
    class_totals = labels.sum(axis=0)
    class_weight = class_totals.max() / class_totals

    (train_x, test_x, train_y, test_y) = train_test_split(
        dataset, labels, test_size=0.20, stratify=labels, random_state=42
    )
    return train_x, test_x, train_y, test_y, class_weight, label_enc.classes_

def checkpoint_call(directory):
    '''Return a callback checkpoint configuration to save only the best model'''
    fname = os.path.sep.join([directory, "weights.hdf5"])
    checkpoint = ModelCheckpoint(
        fname, monitor="val_loss", mode="min", save_best_only=True,
        verbose=1
    )
    return checkpoint

def lenet_training(args, train_x, test_x, train_y, test_y, class_weight):
    '''Launch the lenet training'''
    print("[INFO] Compiling model...")
    model = LeNet.build(width=28, height=28, depth=1, classes=2)
    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
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
    history = model.fit(
        train_x, train_y, validation_data=(test_x, test_y),
        class_weight=class_weight, batch_size=64, epochs=15,
        callbacks=callbacks, verbose=1
    )
    return history, model

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

def display_learning_evol(history_dic, directory):
    '''Plot the training loss and accuracy'''
    fname = os.path.sep.join([directory, "loss_accuracy_history.png"]
    )
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
    dataset, labels = data_loader(args["dataset"])

    train_x, test_x, train_y, test_y, class_weight, lab_name = data_preparation(
        dataset, labels
    )

    history, model = lenet_training(
        args, train_x, test_x, train_y, test_y, class_weight
    )

    model_evaluation(model, test_x, test_y, lab_name)

    display_learning_evol(history, args["output"])


if __name__ == "__main__":
    main()
