#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:37:49 2020

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

from imagetoarraypreprocessor import ImageToArrayPreprocessor
from data_tools import SimplePreprocessor
from data_tools import SimpleDatasetLoader
from shallownet import ShallowNet


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Vegetable Classification",
        usage='''%(prog)s [Application of SwallowNet CNNs structure]''',
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

def main():
    '''Launch main process'''
    args = arguments_parser()


if __name__ == "__main__":
    main()
