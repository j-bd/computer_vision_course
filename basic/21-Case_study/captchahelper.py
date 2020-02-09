#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 17:40:27 2020

@author: j-bd
Method to pad and resize input images to a fixed size without distorting their
aspect ratio
"""

import imutils
import cv2


def preprocess(image, width, height):
    '''Pad and resize input images'''
    # Grab the dimensions of the image, then initialize the padding values
    (image_h, image_w) = image.shape[:2]
    # If the width is greater than the height then resize along the width
    if image_w > image_h:
        image = imutils.resize(image, width=width)
    # Otherwise, the height is greater than the width so resize along the height
    else:
        image = imutils.resize(image, height=height)

    # Determine the padding values for the width and height to obtain the target
    # dimensions
    pad_width = int((width - image.shape[1]) / 2.0)
    pad_height = int((height - image.shape[0]) / 2.0)
    # Pad the image then apply one more resizing to handle any rounding issues
    image = cv2.copyMakeBorder(
        image, pad_height, pad_height, pad_width, pad_width,
        cv2.BORDER_REPLICATE
    )
    image = cv2.resize(image, (width, height))
    # Return the pre-processed image
    return image

def main():
    '''Launch main steps'''


if __name__ == "__main__":
    main()
