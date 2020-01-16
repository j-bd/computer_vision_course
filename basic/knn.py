#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:54:01 2020

@author: j-bd

k-NN classifier on the raw pixel intensities of the food dataset and use it to
classify unknown food image

We use 4 steps pipeline:
    Step #1 – Gather Our Dataset and Preprocess
    Step #2 – Split the Dataset
    Step #3 – Train the Classifier
    Step #4 – Evaluate
"""
import argparse
import logging

from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from data_tools import SimplePreprocessor, SimpleDatasetLoader

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def arguments_parser():
    '''Get the informations from the operator'''
    parser = argparse.ArgumentParser(

        prog='k-NN classifier',
        usage='''
        %(prog)s [on the raw pixel intensities of the food dataset
         and use it to classify unknown food image]
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 main.py --dataset path/to/dataset/folder --neighbors int
        --jobs int

        The following argument is mandatory:
        --dataset: The path to where our input image dataset resides on disk
        The following arguments are optionnals:
        --neighbors: the number of neighbors k to apply when using the
        k-NN algorithm.
        --jobs: the number of concurrent jobs to run when computing
        the distance between an input data point and the training set. A value
        of -1 will use all available cores on the processor.
        '''
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="path to input dataset"
    )
    parser.add_argument(
        "-k", "--neighbors", type=int, default=1,
        help="# of nearest neighbors for classification"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=-1,
        help="# of jobs for k-NN distance (-1 uses all available cores)"
    )
    args = vars(parser.parse_args())
    return args


def main():
    '''Launch the mains steps'''
    args = arguments_parser()

    # Step 1: Gather data and preprocess
    logging.info("Loading images and preprocessing in progress ...")
    imagepaths = list(paths.list_images(args["dataset"]))
    # initialize the image preprocessor, load the dataset from disk,
    # and reshape the data matrix
    sp = SimplePreprocessor(32, 32)
    sdl = SimpleDatasetLoader(preprocessors=[sp])
    (data, labels) = sdl.load(imagepaths, verbose=500)
    logging.info(f'''data shape {data.shape}, labels shape {labels.shape}''')
    # in order to apply the k-NN algorithm, we need to “flatten” our images from
    # a 3D representation to a single list of pixel intensities
    # the flattening is 32 × 32 × 3 = 3072
    data = data.reshape((data.shape[0], 3072))
    # show some information on memory consumption of the images
    logging.info(
        "Features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0))
    )

    # Step 2: Split data
    logging.info("Splitting data in progress ...")
    # encode the labels as integers
    lab_enc = LabelEncoder()
    labels = lab_enc.fit_transform(labels)
    # partition the data into training and testing splits using 90% of
    # the data for training and the remaining 10% for testing
    (trainX, testX, trainY, testY) = train_test_split(
        data, labels, test_size=0.10, random_state=42
    )

    # Step 3: kNN training
    logging.info("Training k-NN classifier in progress ...")
    # train a k-NN classifier on the raw pixel intensities
    # the k-NN model is simply storing the trainX and trainY data internally so
    # it can create predictions on the testing set by computing the distance
    # between the input data and the trainX data
    model = KNeighborsClassifier(
        n_neighbors=args["neighbors"], n_jobs=args["jobs"]
    )
    model.fit(trainX, trainY)

    # Step 4: Evaluate
    logging.info("k-NN classifier evaluation in progress ...")
    print(
        classification_report(
            testY, model.predict(testX), target_names=lab_enc.classes_
        )
    )


if __name__ == "__main__":
    main()
