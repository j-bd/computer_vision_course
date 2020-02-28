#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:34:46 2020

@author: j-bd
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

from data_tools import ImageToArrayPreprocessor, SimpleDatasetLoader
from aspectawarepreprocessor import AspectAwarePreprocessor
from minivggnet_tf import MiniVGGNet




def main():
    '''Launch main steps'''


if __name__ == "__main__":
    main()