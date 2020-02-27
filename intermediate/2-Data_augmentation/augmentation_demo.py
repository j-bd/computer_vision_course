#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 08:56:52 2020

@author: j-bd
"""

import argparse

import numpy as np
from keras.preprocessing.image import ImageDataGenerator # data augmentation
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
        python3 augmentation_demo.py --image "path/to/image"
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

def data_loader(input_image):
    '''Load image'''
    # Load the input image, convert it to a NumPy array, and then reshape it to
    # have an extra dimension
    print("[INFO] Loading example image...")
    image = load_img(input_image)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def image_generator(original_image, args):
    '''Generate different image based on original image'''
    # Construct the image generator for data augmentation then initialize the
    # total number of images generated thus far
    augmentation = ImageDataGenerator(
        rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
        fill_mode="nearest"
    )
    total = 0

    # Construct the actual Python generator
    print("[INFO] Generating images...")
    image_gen = augmentation.flow(
        original_image, batch_size=1, save_to_dir=args["output"],
        save_prefix=args["prefix"], save_format="jpeg"
    )
    # Loop over examples from our image data augmentation generator
    for image in image_gen:
    # Increment our counter
        total += 1
    # If we have reached 10 examples, break from the loop
        if total == 10:
            break

def main():
    '''Launch main process'''
    args = arguments_parser()

    image = data_loader(args["image"])

    image_generator(image, args)


if __name__ == "__main__":
    main()
