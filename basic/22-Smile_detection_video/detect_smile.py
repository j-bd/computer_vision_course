#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:18:31 2020

@author: j-bd
"""

import argparse

import cv2
import imutils
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def arguments_parser():
    '''Retrieve  user data command'''
    parser = argparse.ArgumentParser(
        prog="Smile Detection",
        usage='''%(prog)s [Detection procedure]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch execution:
        -------------------------------------
        python3 detect_smile.py
        --cascade "path/to/cascade/directory" --model "path/to/model"
        --video "path/to/video"

        The two first arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-c", "--cascade", required=True,
        help="path to where the face cascade resides"
    )
    parser.add_argument(
        "-m", "--model", required=True,
        help="path to pre-trained smile detector CNN"
    )
    parser.add_argument(
        "-v", "--video", help="path to the (optional) video file"
    )
    args = vars(parser.parse_args())
    return args

def main():
    '''Launch main steps'''
    args = arguments_parser()


if __name__ == "__main__":
    main()
