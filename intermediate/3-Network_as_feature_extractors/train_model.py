#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:20:48 2020

@author: j-bd
"""

import os
import argparse
import pickle

import h5py
import pandas as pd
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
        "-o", "--output", required=True, help="directory to save model and plot"
    )
    parser.add_argument(
        "-tb", "--tboutput", required=True, help="path to TensorBoard directory"
    )
    args = vars(parser.parse_args())
    return args

def database_loader(db_path):
    '''Load and return the HDF5 database and the value of train data size'''
    # open the HDF5 database for reading then determine the index of
    # the training and testing split, provided that this data was
    # already shuffled *prior* to writing it to disk
    h5_db = h5py.File(db_path, "r")
    train_size = int(h5_db["labels"].shape[0] * 0.75)
    return h5_db, train_size

def log_reg_training(h5_db, train_size, job_numbers):
    '''Train Logistic Regression'''
    # Define the set of parameters that we want to tune then start a
    # grid search where we evaluate our model for each value of C
    print("[INFO] Tuning hyperparameters...")
    params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
    model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=job_numbers)
    model.fit(h5_db["features"][:train_size], h5_db["labels"][:train_size])

    print("[INFO] Best hyperparameters: {}".format(model.best_params_))
    return model

def log_reg_eval(model, h5_db, train_size, directory):
    '''Evaluate the model'''
    print("[INFO] Evaluating...")
    preds = model.predict(h5_db["features"][train_size:])
    report = classification_report(
        h5_db["labels"][train_size:], preds, target_names=h5_db["label_names"]
    )
    print(report)
    dataframe = pd.DataFrame.from_dict(report)
    dataframe.to_csv(
        os.path.sep.join([directory, "classification_report.csv"]), index=False
    )

def main():
    '''Launch main steps'''
    args = arguments_parser()
    h5_db, train_size = database_loader(args["db"])
    model = log_reg_training(h5_db, train_size, args["jobs"])
    log_reg_eval(model, h5_db, train_size, args["output"])


if __name__ == "__main__":
    main()
