#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:53:41 2020

@author: j-bd
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


class MiniVGGNet:
    '''Define MiniVGGNet CNNs architecture'''
    @staticmethod
    def build(width, height, depth, classes):
        '''Initialize the model along with "channels last" as input shape'''
        model = Sequential()
        input_shape = (height, width, depth)
        # Batch normalization operates over the channels, so in order to apply
        # BN, we need to know the index of the channel dimension.
        chan_dim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1

        # First: (CONV => RELU => BN) * 2 => POOL => DO
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        # Since we do not explicitly set a stride, Keras implicitly assumes our
        # stride to be equal to the max pooling size (which is 2 × 2)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # The POOL layer will randomly disconnect from the next layer with a
        # probability of 25% during training
        model.add(Dropout(0.25))

        # Second: (CONV => RELU => BN) * 2 => POOL => DO
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # Return the constructed network architecture
        return model
