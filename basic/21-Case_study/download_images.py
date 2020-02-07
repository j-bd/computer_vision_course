#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:31:34 2020

@author: j-bd
"""

import argparse
import time
import os

import requests


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Case study",
        usage='''%(prog)s [Gather raw data]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 download_images.py.py
        --output "path/to/output/directory"  --nb_images integer
        Output argument is mandatory.
        '''
    )
    parser.add_argument(
        "-o", "--output", required=True, help="path to output directory of images"
    )
    parser.add_argument(
        "-n", "--nb_images", type=int, default=500,
        help="number of images to download"
    )
    args = vars(parser.parse_args())
    return args

def main():
    '''Launch main steps'''
    args = arguments_parser()


if __name__ == "__main__":
    main()
