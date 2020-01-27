#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 15:41:30 2020

@author: j-bd
"""

import argparse
import logging

import numpy as np
from skimage.exposure import rescale_intensity
import cv2

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def arguments_parser():
    '''Get the informations from the operator'''
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(
        prog="Home made convolution",
        usage='''%(prog)s [with different kind of kernels]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 main.py --image path/to/image.jpeg

        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-i", "--image", required=True, help="path to the input image"
    )
    args = vars(parser.parse_args())
    return args

def convolve(image, kernel):
    '''Define the convolve method'''
    # grab the spatial dimensions of the image and kernel
    (im_h, im_w) = image.shape[:2]
    (kernel_h, kernel_w) = kernel.shape[:2]
    # allocate memory for the output image, taking care to "pad" the borders of
    # the input image so the spatial size (i.e., width and height) are not
    # reduced
    pad = (kernel_w - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((im_h, im_w), dtype="float")

    # loop over the input image, "sliding" the kernel across each (x, y)
    #-coordinate from left-to-right and top-to-bottom
    for y_coord in np.arange(pad, im_h + pad):
        for x_coord in np.arange(pad, im_w + pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[
                y_coord - pad:y_coord + pad + 1, x_coord - pad:x_coord + pad + 1
            ]

            #perform the actual convolution by taking the
            #element-wise multiplication between the ROI and
            #the kernel, then summing the matrix
            k = (roi * kernel).sum()

            # store the convolved value in the output (x, y)-
            # coordinate of the output image
            output[y_coord - pad, x_coord - pad] = k

    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    # return the output image
    return output

def kernel_list():
    '''Provide different kide of kernels'''
    # construct average blurring kernels used to smooth an image
    small_blur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
    large_blur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

    # construct a sharpening filter
    sharpen = np.array(
        ([0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]), dtype="int"
    )

    # construct the Laplacian kernel used to detect edge-like regions of an image
    laplacian = np.array(
        ([0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]), dtype="int"
    )

    # The Sobel kernels can be used to detect edge-like regions along both the x
    # and y axis, respectively
    # construct the Sobel x-axis kernel
    sobel_x = np.array(
        ([-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]), dtype="int"
    )
    # construct the Sobel y-axis kernel
    sobel_y = np.array(
        ([-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]), dtype="int"
    )

    # construct an emboss kernel
    emboss = np.array(
        ([-2, -1, 0],
         [-1, 1, 1],
         [0, 1, 2]), dtype="int"
    )

    # construct the kernel bank, a list of kernels we’re going to apply using both
    # our custom ‘convole‘ function and OpenCV’s ‘filter2D‘ function
    kernel_bank = (
        ("small_blur", small_blur),
        ("large_blur", large_blur),
        ("sharpen", sharpen),
        ("laplacian", laplacian),
        ("sobel_x", sobel_x),
        ("sobel_y", sobel_y),
        ("emboss", emboss)
    )
    return kernel_bank

def main():
    '''Launch the mains steps'''
    args = arguments_parser()

    kernel_bank = kernel_list()

    # load the input image and convert it to grayscale
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # loop over the kernels
    for (kernel_name, kernel) in kernel_bank:
        # apply the kernel to the grayscale image using both our custom
        # ‘convolve‘ function and OpenCV’s ‘filter2D‘ function
        print("[INFO] applying {} kernel".format(kernel_name))
        convolve_output = convolve(gray, kernel)
        opencv_output = cv2.filter2D(gray, -1, kernel)

        # show the output images
        cv2.imshow("Original", gray)
        cv2.imshow("{} - convole".format(kernel_name), convolve_output)
        cv2.imshow("{} - opencv".format(kernel_name), opencv_output)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
