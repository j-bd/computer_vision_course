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

def image_preprocess(path):
    '''Greyscale, pad and threshold image'''
    image = cv2.imread(path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    return thresh

def contours_finder(thresh):
    '''Determine contours of elements'''
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = cnts[1] if imutils.is_cv3() else cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
    cnts = contours.sort_contours(cnts)[0]
    # initialize the output image as a "grayscale" image with 3
    # channels along with the output predictions
    output = cv2.merge([gray] * 3)
    predictions = []

def main():
    '''Launch main steps'''
    args = arguments_parser()

    print("[INFO] Loading pre-trained network...")
    model = load_model(args["model"])

    # randomly sample a few of the input images
    image_paths = list(paths.list_images(args["input"]))
    image_paths = np.random.choice(image_paths, size=(10,), replace=False)

    for path in paths:
        thresh = image_preprocess(path)
        cnts = contours_finder(thresh)



if __name__ == "__main__":
    main()
