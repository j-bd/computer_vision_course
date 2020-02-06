#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:20:01 2020

@author: j-bd
"""

import argparse

import cv2
import numpy as np
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY, not with Theano
from keras.applications import VGG16
from keras.applications import VGG19
# Pre-processing our input images and decoding output classifications easier
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Pretrained models",
        usage='''%(prog)s [Images classification]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 imagenet_pretrained.py
        --image "path/to/input/image"  --model "model name to be used"
        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-i", "--image", required=True, help="path to the input image"
    )
    parser.add_argument(
        "-m", "--model", type=str, default="vgg16",
        help="name of pre-trained network to use"
    )
    args = vars(parser.parse_args())
    return args

def main():
    '''Launch main steps'''
    args = arguments_parser()


if __name__ == "__main__":
    main()
