#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:01:39 2020

@author: j-bd
"""

import os

import h5py


class HDF5DatasetWriter:
    '''Class responsible for taking an input set of NumPy arrays (whether
    features, raw images, etc.) and writing them to HDF5 format'''
    def __init__(self, dims, output_path, data_key="images", buf_size=1000):
        '''Initialise writter object'''
        # Check to see if the output path exists, and if so, raise an exception
        if os.path.exists(output_path):
            raise ValueError(
                "The supplied ‘outputPath‘ already exists and cannot be "
                "overwritten. Manually delete the file before continuing.",
                output_path
            )

        # Open the HDF5 database for writing and create two datasets: one to
        # store the images/features and another to store the class labels
        self.dbase = h5py.File(output_path, "w")
        self.data = self.dbase.create_dataset(data_key, dims, dtype="float")
        self.labels = self.dbase.create_dataset(
            "labels", (dims[0],), dtype="int"
        )

        # Store the buffer size, then initialize the buffer itself along with
        # the index into the datasets. "buf_size" controls the size of our in
        #-memory buffer. Once we reach it, we’ll flush the buffer to the HDF5
        # dataset
        self.buf_size = buf_size
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        '''Add new element'''
        # Add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.buf_size:
            self.flush()

    def flush(self):
        '''Write the buffers to disk then reset the buffer'''
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def store_class_labels(self, class_labels):
        '''Create a dataset to store the actual class label names, then store
        the class labels'''
        dset = h5py.special_dtype(vlen=str)
        label_set = self.dbase.create_dataset(
            "label_names", (len(class_labels),), dtype=dset
        )
        label_set[:] = class_labels

    def close(self):
        '''Close the dataset'''
        # check to see if there are any other entries in the buffer that need to
        # be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.dbase.close()
