#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:31:34 2020

@author: j-bd
"""

import argparse
import time
import os
import random

import requests
from PIL import Image, ImageDraw, ImageFont


#URL = "https://www.e-zpassny.com/vector/jcaptcha.do"

def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Case study",
        usage='''%(prog)s [Gather raw data]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 download_images.py
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

def download(args):
    '''download the captcha images available in URL'''
    total = 0

    # loop over the number of images to download
    for i in range(0, args["nb_images"]):
        try:
            # try to grab a new captcha image
            request = requests.get(URL, timeout=60)

            # save the image to disk
            path = os.path.sep.join(
                [args["output"], "{}.jpg".format(str(total).zfill(5))]
            )
            file = open(path, "wb")
            file.write(request.content)
            file.close()

            # update the counter
            print("[INFO] downloaded: {}".format(path))
            total += 1

        # handle if any exceptions are thrown during the download process
        except:
            print("[INFO] error downloading image...")
        # insert a small sleep to be courteous to the server
        time.sleep(0.1)

def create(args):
    '''Create 'jpg' file of 4 numbers'''
    total = 0
    for i in range(0, args["nb_images"]):
        img = Image.new('RGB', (200, 60), color=(255, 255, 255))
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
        numbers = (
            f"{random.randint(0,9)} {random.randint(0,9)} {random.randint(0,9)} "
            f"{random.randint(0,9)}"
        )
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), numbers, font=fnt, fill=(0, 0, 0))
        path = os.path.sep.join(
            [args["output"], "{}.jpg".format(str(total).zfill(5))]
        )
        img.save(path)
        total += 1

def main():
    '''Launch main steps'''
    args = arguments_parser()
    create(args)


if __name__ == "__main__":
    main()
