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
        --model path/to/folder/

        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="path to input dataset"
    )
    parser.add_argument(
        "-m", "--model", required=True, help="path to output model"
    )
    args = vars(parser.parse_args())
    return args

def preprocessing(arg):
    '''Prepare data and labels under numpy format'''
    # grab the list of images that we’ll be describing
    image_paths = list(paths.list_images(arg))

    # initialize the image preprocessors
    resize = SimplePreprocessor(32, 32)
    image_to_array = ImageToArrayPreprocessor()

    # load the dataset from disk then scale the raw pixel intensities
    # to the range [0, 1]
    dataset_loader = SimpleDatasetLoader(preprocessors=[resize, image_to_array])
    (data, labels) = dataset_loader.load(image_paths, verbose=50)
    data = data.astype("float") / 255.0
    return data, labels

def main():
    '''Launch main process'''
    args = arguments_parser()

    print(" Loading and preprocessing images...")
    data, labels = preprocessing(args["dataset"])

    print(" Splitting data...")
    (train_x, test_x, train_y, test_y) = train_test_split(
        data, labels, test_size=0.25, random_state=42
    )

    # convert the labels from integers to vectors
    train_y = LabelBinarizer().fit_transform(train_y)
    test_y = LabelBinarizer().fit_transform(test_y)


if __name__ == "__main__":
    main()
