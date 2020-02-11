#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:08:27 2020

@author: j-bd
"""

import os
import argparse

import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import np_utils
from imutils import paths

from lenet import LeNet



def main():
    '''Launch main steps'''


if __name__ == "__main__":
    main()
