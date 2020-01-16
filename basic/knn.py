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

from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from data_tools import SimplePreprocessor, SimpleDatasetLoader

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

if __name__ == "__main__":
    main()
