#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:44:27 2020

@author: j-bd
"""

from keras.utils import plot_model

from lenet import LeNet


def main():
    '''Launch main steps'''
    # initialize LeNet and then write the network architecture visualization
    # graph to disk
    model = LeNet.build(28, 28, 1, 10)
    plot_model(model, to_file="lenet.png", show_shapes=True)



if __name__ == "__main__":
    main()
