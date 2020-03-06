#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:20:54 2020

@author: j-bd
"""

import os
import argparse
import random

import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
import progressbar
from imutils import paths

from hdf5datasetwriter import HDF5DatasetWriter



def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Feature Extraction Process",
        usage='''%(prog)s [VGG16]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 .py
        --dataset "path/to/dataset/directory" --output "path/to/model/directory"
        --tboutput "path/to/directory"
        --batch-size 32 --buffer-size 1000

        The three first arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="path to input dataset"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="directory to save model and plot"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=32,
        help="batch size of images to be passed through network"
    )
    parser.add_argument(
        "-s", "--buffer-size", type=int, default=1000,
        help="size of feature extraction buffer"
    )
    parser.add_argument(
        "-tb", "--tboutput", required=True, help="path to TensorBoard directory"
    )
    args = vars(parser.parse_args())
    return args

def data_loader(data_directory):
    '''Load data and corresponding labels from disk'''
    # Grab the list of images that we’ll be describing then randomly shuffle
    # them to allow for easy training and testing splits via array slicing
    # during training time
    print("[INFO] Loading images...")
    image_paths = list(paths.list_images(data_directory))
    random.shuffle(image_paths)

    # Extract the class labels from the image paths then encode the labels
    labels = [path.split(os.path.sep)[-2] for path in image_paths]
    lab_enc = LabelEncoder()
    labels = lab_enc.fit_transform(labels)

    return image_paths, labels, lab_enc.classes_

def features_extr(image_paths, args, labels, lab_classes):
    '''Extract features from VGG16 network'''
    # Load the VGG16 network without he final fully-connected layers
    print("[INFO] Loading network...")
    model = VGG16(weights="imagenet", include_top=False)

    # Initialize the HDF5 dataset writer, then store the class label names in
    # the dataset
    dataset = HDF5DatasetWriter(
        (len(image_paths), 512 * 7 * 7), args["output"], data_key="features",
        buf_size=args["buffer_size"]
    )
    dataset.store_class_labels(lab_classes)

    # Initialize the progress bar
    widgets = [
        "Extracting Features: ", progressbar.Percentage(), " ",
        progressbar.Bar(), " ", progressbar.ETA()
    ]
    pbar = progressbar.ProgressBar(
        maxval=len(image_paths), widgets=widgets
    ).start()

    # Loop over the images in patches
    batch_s = args["batch-size"]
    for i in np.arange(0, len(image_paths), batch_s):
        # Extract the batch of images and labels, then initialize the list of
        # actual images that will be passed through the network for feature
        # extraction
        batch_paths = image_paths[i:i + batch_s]
        batch_labels = labels[i:i + batch_s]
        batch_images = []

        # Loop over the images and labels in the current batch
        for (j, image_path) in enumerate(batch_paths):
            # load the input image using the Keras helper utility
            # while ensuring the image is resized to 224x224 pixels
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)

            # Preprocess the image by (1) expanding the dimensions and (2)
            # subtracting the mean RGB pixel intensity from the ImageNet dataset
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)

            # Add the image to the batch
            batch_images.append(image)

        # To obtain our feature vectors pass the images through the network and use the outputs as
        # our actual features
        batch_images = np.vstack(batch_images)
        features = model.predict(batch_images, batch_size=batch_s)

        # reshape the features so that each image is represented by
        # a flattened feature vector of the ‘MaxPooling2D‘ outputs
        features = features.reshape((features.shape[0], 512 * 7 * 7))

        # add the features and labels to our HDF5 dataset
        dataset.add(features, batch_labels)
        pbar.update(i)

    # close the dataset
    dataset.close()
    pbar.finish()

def main():
    '''Launch main steps'''
    args = arguments_parser()
    image_paths, labels, lab_classes = data_loader(args["dataset"])
    features_extr(image_paths, args, labels, lab_classes)


if __name__ == "__main__":
    main()
