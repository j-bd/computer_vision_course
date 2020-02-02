#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:31:27 2020

@author: j-bd
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import BaseLogger


class TrainingMonitor(BaseLogger):
    '''Class called at the end of every epoch when training a network with Keras'''
    def __init__(self, figPath, jsonPath=None, startAt=0):
        '''Initialize the class'''
        # store the output path for the figure, the path to the JSON serialized
        # file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
