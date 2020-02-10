#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:07:38 2020

@author: j-bd
"""

import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from lenet import LeNet
from captchahelper import preprocess


def main():
    '''Launch main steps'''


if __name__ == "__main__":
    main()