#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:32:21 2020

@author: j-bd
"""

import argparse

import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
from imutils import contours
from imutils import paths

from captchahelper import preprocess


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Case study",
        usage='''%(prog)s [Testing CNNs]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 train_model.py
        --input "path/to/dataset/directory" --model "path/to/output/model.hdf5"

        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-i", "--input", required=True, help="path to input dataset"
    )
    parser.add_argument(
        "-m", "--model", required=True, help="path to input model"
    )
    args = vars(parser.parse_args())
    return args

def main():
    '''Launch main steps'''
    args = arguments_parser()


if __name__ == "__main__":
    main()
