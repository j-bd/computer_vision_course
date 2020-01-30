#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:43:37 2020

@author: j-bd
"""
import argparse

from imutils import paths
import numpy as np
import cv2
from keras.models import load_model

from data_tools import ImageToArrayPreprocessor
from data_tools import SimplePreprocessor
from data_tools import SimpleDatasetLoader


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Serialization",
        usage='''%(prog)s [Loading]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 shallownet_train.py --dataset path/to/folder/containing_image
        --model path/to/folder/weights.hdf5

        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="path to input dataset"
    )
    parser.add_argument(
        "-m", "--model", required=True, help="path to pre-trained model"
    )
    args = vars(parser.parse_args())
    return args

def preprocessing(arg):
    '''Prepare data and labels under numpy format'''
    # Grab the list of images in the dataset then randomly sample indexes into
    # the image paths list
    print("INFO: Sampling images...")
    image_paths = np.array(list(paths.list_images(arg)))
    idxs = np.random.randint(0, len(image_paths), size=(10,))
    image_paths = image_paths[idxs]

    # Initialize the image preprocessors
    resize = SimplePreprocessor(32, 32)
    image_to_array = ImageToArrayPreprocessor()

    # Load the dataset from disk then scale the raw pixel intensities to the
    # range [0, 1]
    dataset_loader = SimpleDatasetLoader(preprocessors=[resize, image_to_array])
    (data, labels) = dataset_loader.load(image_paths)
    data = data.astype("float") / 255.0
    return data, labels, image_paths

def main():
    '''Launch main process'''
    args = arguments_parser()

    # Initialize the class labels
    class_labels = ["fries", "beans", "potatoes"]

    data, labels, image_paths = preprocessing(args["dataset"])

    # Load the pre-trained network
    print("INFO: Loading pre-trained network...")
    model = load_model(args["model"])

    # Make predictions on the images
    # '.predict' method of model will return a list of probabilities for every
    # image in data – one probability for each class label, respectively. Taking
    # the argmax on axis=1 finds the index of the class label with the largest
    # probability for each image.
    print("INFO: Predicting...")
    predictions = model.predict(data, batch_size=32).argmax(axis=1)

    # Loop over the sample images
    for (i, image_path) in enumerate(image_paths):
        # Load the example image, draw the prediction, and display it to our
        # screen
        image = cv2.imread(image_path)
        cv2.putText(image, "Label: {}".format(class_labels[predictions[i]]),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Image", image)
        cv2.waitKey(10000)


if __name__ == "__main__":
    main()
