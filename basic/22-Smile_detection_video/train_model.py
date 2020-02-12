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
        --dataset "path/to/dataset/directory" --model "path/to/output/model.hdf5"
        --weights "path/to/weights/directory" --tboutput "path/to/directory"
        --history "path/to/directory/history.png"
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
        "-w", "--weights", required=True, help="path to weights directory"
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
    return train_x, test_x, train_y, test_y, class_weight

def main():
    '''Launch main steps'''
    args = arguments_parser()
    dataset, labels = data_loader(args["dataset"])

    train_x, test_x, train_y, test_y, class_weight = data_preparation(
        dataset, labels
    )


if __name__ == "__main__":
    main()
