#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:44:31 2020

@author: j-bd
"""

import os
import argparse

import matplotlib
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from keras.datasets import cifar10

from trainingmonitor import TrainingMonitor
from minivggnet import MiniVGGNet
