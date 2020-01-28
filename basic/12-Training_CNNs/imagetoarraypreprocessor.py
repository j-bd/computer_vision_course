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
    def __init__(self, data_format=None):
        '''Initialize elements for preprocessing'''
        # store the image data format. This value defaults to None, which
        # indicates that the setting inside keras.json should be used
        self.data_format = data_format

    def preprocess(self, image):
        '''Apply the Keras utility function that correctly rearranges the
        dimensions of the image. Returns a new NumPy array with the channels
        properly ordered'''
        return img_to_array(image, data_format=self.data_format)
