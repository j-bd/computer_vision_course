#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 08:56:52 2020

@author: j-bd
"""

import argparse

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


def arguments_parser():
    '''Retrieve user commands'''
    parser = argparse.ArgumentParser(
        prog="Augmentation data",
        usage='''%(prog)s [Computer Vision courses]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch execution:
        -------------------------------------
        python3 augmentation_demo.py --image "path/to/image/directory"
        --output "path/to/output/directory" --prefix "filename prefix"

        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-i", "--image", required=True, help="Path to the image"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="path to output directory to store augmentation examples"
    )
    parser.add_argument(
        "-p", "--prefix", type=str, default="image",
        help="output filename prefix"
    )
    args = vars(parser.parse_args())
    return args

def main():
    '''Launch main process'''
    args = arguments_parser()


if __name__ == "__main__":
    main()
