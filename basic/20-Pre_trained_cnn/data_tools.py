#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:56:44 2020

@author: j-bd
"""

import os

import numpy as np
import cv2
from keras.preprocessing.image import img_to_array


class SimplePreprocessor:
    '''Preprocess images'''
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        '''Store the target image width, height, and interpolation
        Method used when resizing'''
        self.width = width
        self.height = height
        #control which interpolation algorithm is used when resizing
        self.inter = inter

    def preprocess(self, image):
        '''Resize the image to a fixed size, ignoring the aspect ratio'''
        return cv2.resize(
            image, (self.width, self.height), interpolation=self.inter
        )


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


class SimpleDatasetLoader:
    '''Load image'''
    def __init__(self, preprocessors=None):
        '''Store the image preprocessor'''
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            # We declare as a list because sometimes we want to take a sequence
            # of preproceesing actions : resize, scalling, converting. Thus, we
            # have independently implementation
            self.preprocessors = list()

    def load(self, imagepaths, verbose=-1):
        '''images and labels loading'''
        # Initialize the list of features/images and labels
        data = list()
        labels = list()

        # Loop over input image
        for i, imagepath in enumerate(imagepaths):
            # load the image and extract the class label assuming the path has
            # the following format: /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagepath)
            label = imagepath.split(os.path.sep)[-2]
            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to the image
                for preprocessor in self.preprocessors:
                    image = preprocessor.preprocess(image)
            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            # !! assumes that all images in the dataset can fit into main memory
            # !! at once
            data.append(image)
            labels.append(label)
            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("processed {}/{}".format(i + 1, len(imagepaths)))

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))
