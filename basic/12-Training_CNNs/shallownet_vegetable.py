#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:37:49 2020

@author: j-bd
"""

import argparse
import logging

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD

from imagetoarraypreprocessor import ImageToArrayPreprocessor
from data_tools import SimplePreprocessor
from data_tools import SimpleDatasetLoader
from shallownet import ShallowNet

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Vegetable Classification",
        usage='''%(prog)s [Application of ShallowNet CNNs structure]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 shallownet_vegetable.py --dataset path/to/folder/containing_image

        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="path to input dataset"
    )
    args = vars(parser.parse_args())
    return args

def preprocessing(args):
    '''Prepare data and labels'''
    # grab the list of images that we’ll be describing
    logging.info(" Loading images...")
    image_paths = list(paths.list_images(args["dataset"]))

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
    data, labels = preprocessing(args)



if __name__ == "__main__":
    main()
