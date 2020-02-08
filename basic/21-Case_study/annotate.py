#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:47:54 2020

@author: j-bd
"""

import argparse
import os

import imutils
from imutils import paths
import cv2


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Case study",
        usage='''%(prog)s [Annotate data]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 annotate.py
        --input "path/to/input/directory"
        --annot "path/to/output/annotation/directory"
        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-i", "--input", required=True, help="path to input directory of images"
    )
    parser.add_argument(
        "-a", "--annot", required=True,
        help="path to output directory of annotations"
    )
    args = vars(parser.parse_args())
    return args

def main():
    '''Launch main steps'''
    args = arguments_parser()


if __name__ == "__main__":
    main()
