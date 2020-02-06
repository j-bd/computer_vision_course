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

    # Load the input image using the Keras helper utility while ensuring the
    # image is resized to ‘inputShape‘, the required input dimensions for the
    # ImageNet pre-trained network
    print("[INFO] loading and pre-processing image...")
    image = load_img(args["image"], target_size=input_shape)
    image = img_to_array(image)

    # Input image is now represented as a NumPy array of shape
    # (inputShape[0], inputShape[1], 3) however we need to expand the dimension
    # by making the shape (1, inputShape[0], inputShape[1], 3) so we can pass it
    # through the network

    image = np.expand_dims(image, axis=0)

    # Pre-process the image using the appropriate function based on the model
    # that has been loaded (i.e., mean subtraction, scaling, etc.)
    image = preprocess(image)

    return image

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

def prediction(model, image, args):
    '''Classify the image'''
    print("[INFO] Classifying image with ’{}’...".format(args["model"]))
    preds = model.predict(image)
    result = imagenet_utils.decode_predictions(preds)

    # loop over the predictions and display the rank-5 predictions +
    # probabilities to our terminal
    for (i, (imagenetID, label, prob)) in enumerate(result[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

    return result

def display_result(result, args):
    # load the image via OpenCV, draw the top prediction on the image,
    # and display the image to our screen
    original = cv2.imread(args["image"])
    (imagenetID, label, prob) = result[0][0]
    cv2.putText(original, "Label: {}".format(label), (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Classification", original)
    cv2.waitKey(5000)

def main():
    '''Launch main steps'''
    args = arguments_parser()
    check_input(args)

    image = image_preprocess(args)

    model = model_loading(args)

    result = prediction(model, image, args)

    display_result(result, args)


if __name__ == "__main__":
    main()
