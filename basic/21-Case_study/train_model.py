#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:07:38 2020

@author: j-bd
"""

import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from lenet import LeNet
from captchahelper import preprocess


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Case study",
        usage='''%(prog)s [Training LeNet CNNs]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 train_model.py
        --dataset "path/to/dataset/directory" --model "path/to/output/model"
        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-", "--dataset", required=True, help="path to input dataset"
    )
    parser.add_argument(
        "-m", "--model", required=True, help="path to output model"
    )
    args = vars(parser.parse_args())
    return args

def main():
    '''Launch main steps'''
    args = arguments_parser()


if __name__ == "__main__":
    main()