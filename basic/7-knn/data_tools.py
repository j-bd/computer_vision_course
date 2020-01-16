#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:24:21 2020

@author: j-bd
"""
import os
import logging

import numpy as np
import cv2


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

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
                print("i")
                logging.info("processed {}/{}".format(i + 1, len(imagepaths)))

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))
