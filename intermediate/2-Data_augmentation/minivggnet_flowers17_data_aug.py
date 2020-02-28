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

def main():
    '''Launch main steps'''
    args = arguments_parser()

    image_paths, cl_labels = data_loader(args["dataset"])




if __name__ == "__main__":
    main()