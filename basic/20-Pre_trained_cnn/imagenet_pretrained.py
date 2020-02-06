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


MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}

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
        Image argument is mandatory.
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

def check_input(args):
    '''Check if model argument is in MODELS dictionary'''
    if args["model"] not in MODELS.keys():
        raise AssertionError(
            "The --model command line argument should be a key in the ‘MODELS‘ "\
            "dictionary"
        )

def image_preprocess(args):
    '''Provide image resizing preprocessing'''
    # Initialize the input image shape (224x224 pixels) along with the
    # pre-processing function (this might need to be changed based on which
    # model we use to classify our image)
    input_shape = (224, 224)

    preprocess = imagenet_utils.preprocess_input

    # If we are using the InceptionV3 or Xception networks, then we
    # need to set the input shape to (299x299) [rather than (224x224)]
    # and use a different image processing function
    if args["model"] in ("inception", "xception"):
        input_shape = (299, 299)
        preprocess = preprocess_input
    return input_shape, preprocess

def model_loading(args):
    '''Load our the network weights from disk'''
    # (NOTE: if this is the first time you are running this script for a given
    # network, the weights will need to be downloaded first -- depending on
    # which network you are using, the weights can be 90-575MB, so be patient;
    # the weights will be cached and subsequent runs of this script will be
    # *much* faster)
    print("[INFO] Loading {}...".format(args["model"]))
    network = MODELS[args["model"]]
    model = network(weights="imagenet")

    return model

def main():
    '''Launch main steps'''
    args = arguments_parser()
    check_input(args)

    input_shape, preprocess = image_preprocess(args)

    model = model_loading(args)


if __name__ == "__main__":
    main()
