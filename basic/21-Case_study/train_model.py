#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:07:38 2020

@author: j-bd
"""

import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

from lenet import LeNet
from captchahelper import preprocess


def arguments_parser():
    '''Retrieve user data command'''
    parser = argparse.ArgumentParser(
        prog="Case study",
        usage='''%(prog)s [Training LeNet CNNs]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch custom training execution:
        -------------------------------------
        python3 train_model.py
        --dataset "path/to/dataset/directory" --model "path/to/output/model.hdf5"
        --weights "path/to/weights/directory" --tboutput "path/to/directory"
        --history "path/to/directory/history.png"
        All arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="path to input dataset"
    )
    parser.add_argument(
        "-m", "--model", required=True, help="path to save 'file.hdf5' model"
    )
    parser.add_argument(
        "-w", "--weights", required=True, help="path to weights directory"
    )
    parser.add_argument(
        "-tb", "--tboutput", required=True, help="path to TensorBoard directory"
    )
    parser.add_argument(
        "-h", "--history", required=True, help="path to save 'history.png' model"
    )
    args = vars(parser.parse_args())
    return args

def data_loader(data_directory):
    '''Load data and corresponding labels from disk'''
    data = []
    labels = []

    for image_path in paths.list_images(data_directory):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = preprocess(image, 28, 28)
        image = img_to_array(image)
        data.append(image)

        label = image_path.split(os.path.sep)[-2]
        labels.append(label)
    return data, labels

def data_preparation(dataset, labels):
    '''Scaling and binarize data'''
    dataset = np.array(dataset, dtype="float") / 255.0
    labels = np.array(labels)

    (train_x, test_x, train_y, test_y) = train_test_split(
        dataset, labels, test_size=0.25, random_state=42
    )

    # Convert the labels from integers to vectors
    label_bi = LabelBinarizer()
    train_y = label_bi.fit_transform(train_y)
    test_y = label_bi.fit_transform(test_y)

    return train_x, test_x, train_y, test_y, label_bi

def checkpoint_call(args):
    '''Return a callback checkpoint configuration'''
    # construct the callback to save only the *best* model to disk based on the
    # validation loss
    checkpoint = ModelCheckpoint(
        args, monitor="val_loss", mode="min", save_best_only=True,
        verbose=1
    )
    return checkpoint

def lenet_training(args, train_x, test_x, train_y, test_y):
    '''Launch the lenet training'''
    print("[INFO] Compiling model...")
    model = LeNet.build(width=28, height=28, depth=1, classes=9)
    opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    model.summary()

    # Callbacks creation
    checkpoint_save = checkpoint_call(args["weights"])
    tensor_board = TensorBoard(
        log_dir=args["tboutput"], histogram_freq=1, write_graph=True,
        write_images=True
    )
    callbacks = [checkpoint_save, tensor_board]

    print("[INFO] Training network...")
    history = model.fit(
        train_x, train_y, validation_data=(test_x, test_y), batch_size=32,
        epochs=15, callbacks=callbacks, verbose=1
    )
    model.save(args["model"])

    return history, model

def model_evaluation(model, test_x, test_y, label_names):
    '''Display on terminal command the quality of model's predictions'''
    print("[INFO] Evaluating network...")
    predictions = model.predict(test_x, batch_size=64)
    print(
        classification_report(
            test_y.argmax(axis=1), predictions.argmax(axis=1),
            target_names=label_names
        )
    )

def display_learning_evol(history_dic, saving_path):
    '''Plot the training loss and accuracy'''
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(
        np.arange(0, len(history_dic.history["loss"])),
        history_dic.history["loss"], label="train_loss"
    )
    plt.plot(
        np.arange(0, len(history_dic.history["val_loss"])),
        history_dic.history["val_loss"], label="val_loss"
    )
    plt.plot(
        np.arange(0, len(history_dic.history["accuracy"])),
        history_dic.history["accuracy"], label="train_acc"
    )
    plt.plot(
        np.arange(0, len(history_dic.history["val_accuracy"])),
        history_dic.history["val_accuracy"], label="val_accuracy"
    )
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(saving_path)

def main():
    '''Launch main steps'''
    args = arguments_parser()
    dataset, labels = data_loader(args["dataset"])

    train_x, test_x, train_y, test_y, label_bi = data_preparation(
        dataset, labels
    )

    history, model = lenet_training(args, train_x, test_x, train_y, test_y)

    model_evaluation(model, test_x, test_y, label_bi)

    display_learning_evol(history, args["history"])


if __name__ == "__main__":
    main()