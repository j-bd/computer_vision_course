#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:23:41 2020

@author: j-bd
"""

from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    '''We wrap img_to_array function inside a new classwith a special preprocess
    function'''
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat