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

def processing(args):
    '''Prepare and process images'''
    # Grab the image paths then initialize the dictionary of character counts
    image_paths = list(paths.list_images(args["input"]))
    counts = {}
    # Loop over the image paths
    for (i, image_path) in enumerate(image_paths):
        # Display an update to the user
        print("[INFO] Processing image {}/{}".format(i + 1, len(image_paths)))
        try:
            # Load the image and convert it to grayscale, then pad the image to
            # ensure digits caught on the border of the image are retained
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

            # Threshold the image to reveal the digits. It is a critical step in
            # our image processing pipeline
            thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]

            # Find contours in the image, keeping only the four largest ones
            # just in case there is “noise” in the image
            cnts = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cnts = cnts[1] if imutils.is_cv3() else cnts[0] # Check OpenCV version
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

            # loop over the contours
            for c in cnts:
                # compute the bounding box for the contour then extract the digit
                (x, y, w, h) = cv2.boundingRect(c)
                roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

                # display the character, making it larger enough for us
                # to see, then wait for a keypress
                cv2.imshow("ROI", imutils.resize(roi, width=28))
                key = cv2.waitKey(0)

                # if the ’‘’ key is pressed, then ignore the character
                if key == ord("‘"):
                    print("[INFO] ignoring character")
                    continue
                # grab the key that was pressed and construct the path
                # the output directory
                key = chr(key).upper()
                dirPath = os.path.sep.join([args["annot"], key])

                # if the output directory does not exist, create it
                if not os.path.exists(dirPath):
                    os.makedirs(dirPath)

                # write the labeled character to file
                count = counts.get(key, 1)
                p = os.path.sep.join([dirPath, "{}.png".format(
                str(count).zfill(6))])
                cv2.imwrite(p, roi)
                # increment the count for the current key
                counts[key] = count + 1

        # we are trying to control-c out of the script, so break from the
        # loop (you still need to press a key for the active window to
        # trigger this)
        except KeyboardInterrupt:
            print("[INFO] manually leaving script")
            break
        # an unknown error has occurred for this particular image
        except:
            print("[INFO] skipping image...")

def main():
    '''Launch main steps'''
    args = arguments_parser()
    processing(args)


if __name__ == "__main__":
    main()
