#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:20:54 2020

@author: j-bd
"""

import os
import argparse
import random

from imutils import paths
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
import progressbar

from hdf5datasetwriter import HDF5DatasetWriter


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="",
        usage='''%(prog)s []''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 .py
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
        "-b", "--batch-size", type=int, default=32,
        help="batch size of images to be passed through network"
    )
    parser.add_argument(
        "-s", "--buffer-size", type=int, default=1000,
        help="size of feature extraction buffer"
    )
    parser.add_argument(
        "-tb", "--tboutput", required=True, help="path to TensorBoard directory"
    )
    args = vars(parser.parse_args())
    return args

def main():
    '''Launch main steps'''
    args = arguments_parser()


if __name__ == "__main__":
    main()
