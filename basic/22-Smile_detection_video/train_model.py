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

from lenet import LeNet


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Case study",
        usage='''%(prog)s [Training LeNet CNNs]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 train_model.py
        --dataset "path/to/dataset/directory" --model "path/to/model/directory"
        --tboutput "path/to/directory" --history "path/to/directory/history.png"

        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="path to input dataset"
    )
    parser.add_argument(
        "-m", "--model", required=True, help="path to save 'file.hdf5' model"
    )
    parser.add_argument(
        "-tb", "--tboutput", required=True, help="path to TensorBoard directory"
    )
    parser.add_argument(
        "-hy", "--history", required=True, help="path to save 'history.png' model"
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
    fname = os.path.sep.join(
        [directory, "weights.hdf5"]
    )
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
    checkpoint_save = checkpoint_call(args["model"])
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


if __name__ == "__main__":
    main()
