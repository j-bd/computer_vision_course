#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:42:54 2020

@author: j-bd

Aspect-aware preprocessor is a two step algorithm:
1. Step #1: Determine the shortest dimension and resize along it.
2. Step #2: Crop the image along the largest dimension to obtain the target
width and height.
"""

import cv2
import imutils


class AspectAwarePreprocessor:
    '''Preprocessing applied before deep learning'''
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        '''Store the target image width, height, and interpolation method used
        when resizing'''
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        '''Grab the dimensions of the image and then initialize the deltas to
        use when cropping'''
        (im_h, im_w) = image.shape[:2]
        delta_w = 0
        delta_h = 0
        # if the width is smaller than the height, then resize along the width
        # (i.e., the smaller dimension) and then update the deltas to crop the
        # height to the desired dimension
        if im_w < im_h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            delta_h = int((image.shape[0] - self.height) / 2.0)
        # Otherwise, the height is smaller than the width so resize along the
        # height and then update the deltas to crop along the width
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            delta_w = int((image.shape[1] - self.width) / 2.0)

        # Now that our images have been resized, we need to re-grab the width
        # and height, followed by performing the crop
        (im_h, im_w) = image.shape[:2]
        image = image[delta_h:im_h - delta_h, delta_w:im_w - delta_w]

        # Finally, resize the image to the provided spatial dimensions to ensure
        # our output image is always a fixed size
        return cv2.resize(
            image, (self.width, self.height), interpolation=self.inter
        )
