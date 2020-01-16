#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:24:21 2020

@author: latitude
"""
import os

import numpy as np
import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        '''Store the target image width, height, and interpolation
        Method used when resizing'''
        self.width = width
        self.height = height
        #control which interpolation algorithm is used when resizing
        self.inter = inter

    def preprocess(self, image):
        '''Resize the image to a fixed size, ignoring the aspect r
        atio'''
        return cv2.resize(
            image, (self.width, self.height), interpolation=self.inter
        )



