#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:20:48 2020

@author: j-bd
"""

import argparse
import pickle

import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report



def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="",
        usage='''%(prog)s []''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 .py
        --dataset "path/to/dataset/directory" --output "path/to/model/directory"
        --tboutput "path/to/directory"

        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-db", "--database", required=True, help="path HDF5 database"
    )
    parser.add_argument(
        "-m", "--model", required=True, help="path to output model"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=-1,
        help="numbers of jobs to run when tuning hyperparameters"
    )
    parser.add_argument(
        "-tb", "--tboutput", required=True, help="path to TensorBoard directory"
    )
    args = vars(parser.parse_args())
    return args

def main():
    '''Launch main steps'''
    args = arguments_parser()


if __name__ == "__main__":
    main()
