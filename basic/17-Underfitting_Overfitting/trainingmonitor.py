#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:31:27 2020

@author: j-bd
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import BaseLogger


class TrainingMonitor(BaseLogger):
    '''Class called at the end of every epoch when training a network with Keras
    We extend Keras’ BaseLogger class'''
    def __init__(self, fig_path, json_path=None, start_at=0):
        '''Initialize the class'''
        # Store the output path for the figure, the path to the JSON serialized
        # file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        # The path to the output loss and accuracy plot
        self.fig_path = fig_path
        # Path used to serialize the loss and accuracy values to create custom
        # plots of your own)
        self.json_path = json_path
        # Starting epoch that training is resumed at when using ctrl + c training
        self.start_at = start_at
        # Initialize the history dictionary
        self.loss_history = {}

    def on_train_begin(self, logs={}):
        '''Retrieve or not loss values'''
        # If the JSON history path exists, load the training history
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.loss_history = json.loads(open(self.json_path).read())

                # Check to see if a starting epoch was supplied
                if self.start_at > 0:
                    # Loop over the entries in the history log and trim any
                    # entries that are past the starting epoch
                    for key in self.loss_history.keys():
                        self.loss_history[key] = self.loss_history[key][:self.start_at]

    def on_epoch_end(self, epoch, logs={}):
        '''Process at the end of epoch'''
        # Loop over the logs and update the loss, accuracy, etc.. for the entire
        # training process
        for (key, value) in logs.items():
            log = self.loss_history.get(key, [])
            log.append(value)
            self.loss_history[key] = log

        # Check to see if the training history should be serialized to file
        if self.json_path is not None:
            file = open(self.json_path, "w")
            file.write(json.dumps(self.loss_history))
            file.close()

        # Ensure at least two epochs have passed before plotting
        # (epoch starts at zero)
        if len(self.loss_history["loss"]) > 1:
        # Plot the training loss and accuracy
            x_size = np.arange(0, len(self.loss_history["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(x_size, self.loss_history["loss"], label="train_loss")
            plt.plot(x_size, self.loss_history["val_loss"], label="val_loss")
            plt.plot(
                x_size, self.loss_history["accuracy"], label="train_accuracy"
            )
            plt.plot(
                x_size, self.loss_history["val_accuracy"], label="val_accuracy"
            )
            plt.title(
                "Training Loss and Accuracy [Epoch {}]".format(
                    len(self.loss_history["loss"])
                )
            )
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # Save the figure
            plt.savefig(self.fig_path)
            plt.close()
