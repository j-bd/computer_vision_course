#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:31:03 2020

@author: j-bd

INPUT => CONV => RELU => FC
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:
    '''Define ShallowNet CNNs architecture'''
    @staticmethod
    def build(width, height, depth, classes):
        '''Initialize the model along with "channels last" as input shape'''
        model = Sequential()
        input_shape = (height, width, depth)

        # to be include for nearly every CNN
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
