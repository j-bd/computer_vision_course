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
    return gray, thresh

def contours_finder(thresh):
    '''Determine contours of elements'''
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = cnts[1] if imutils.is_cv3() else cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
    cnts = contours.sort_contours(cnts)[0]
    return cnts

def display_setup(x_val, y_val, w_val, h_val, output, pred):
    '''Prepare the display'''
    # Draw bounding box
    cv2.rectangle(
        output, (x_val - 2, y_val - 2), (x_val + w_val + 4, y_val + h_val + 4),
        (0, 255, 0), 1
    )
    # Draw the predicted digit on the output image itself
    cv2.putText(output, str(pred), (x_val - 5, y_val - 5),
    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

def numbers_prediction(cnts, gray, output, model):
    '''Predict numbers on an image and display the result'''
    predictions = []
    for cnt in cnts:
        # Compute the bounding box for the contour then extract the digit
        (x_val, y_val, w_val, h_val) = cv2.boundingRect(cnt)
        roi = gray[y_val - 5:y_val + h_val + 5, x_val - 5:x_val + w_val + 5]

        # pre-process the ROI and classify it then classify it
        roi = preprocess(roi, 28, 28)
        roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
        pred = model.predict(roi).argmax(axis=1)[0] + 1
        predictions.append(str(pred))

        display_setup(x_val, y_val, w_val, h_val, output, pred)
    # Show the output image
    print("[INFO] Captcha: {}".format("".join(predictions)))
    cv2.imshow("Output", output)
    cv2.waitKey()

def main():
    '''Launch main steps'''
    args = arguments_parser()

    print("[INFO] Loading pre-trained network...")
    model = load_model(args["model"])

    # randomly sample a few of the input images
    image_paths = list(paths.list_images(args["input"]))
    image_paths = np.random.choice(image_paths, size=(10,), replace=False)

    for path in paths:
        gray, thresh = image_preprocess(path)
        cnts = contours_finder(thresh)

        output = cv2.merge([gray] * 3)

        numbers_prediction(cnts, gray, output, model)


if __name__ == "__main__":
    main()
