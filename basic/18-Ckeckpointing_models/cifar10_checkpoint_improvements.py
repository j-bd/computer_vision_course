#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:40:02 2020

@author: j-bd
"""

import os
import argparse

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint

from minivggnet import MiniVGGNet


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Create model checkpoints",
        usage='''%(prog)s [MiniVGGNet on Cifar10 dataset]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 cifar10_monitor.py --weights "path/to/weights/directory"
        --model "path/to/directory/weights.hdf5" --output "path/to/directory"
        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-w", "--weights", required=True,help="path to weights directory"
    )
    parser.add_argument(
        "-m", "--model", required=True, help="path to save model"
    )
    parser.add_argument("-o", "--output", required=True, help="path to output")
    args = vars(parser.parse_args())
    return args

def main():
    '''Launch main steps'''
    args = arguments_parser()


if __name__ == "__main__":
    main()
