#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:34:46 2020

@author: j-bd
"""

import os
import argparse

import numpy as np
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

def main():
    '''Launch main steps'''
    args = arguments_parser()

    image_paths, cl_labels = data_loader(args["dataset"])

    train_x, test_x, train_y, test_y, img_augm_gene = data_preparation(
        image_paths
    )


if __name__ == "__main__":
    main()