#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:38:12 2019

@author: routhier
"""

from keras.models import Sequential
from keras.layers import Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Dense, Conv2D, MaxPooling2D


def cnn_try(num_classes=1):
    """
    """
    window = 2001

    fashion_model = Sequential()

    fashion_model.add(Conv2D(32, kernel_size=(20, 1),
                             activation='exponential',
                             input_shape=(window, 1, 4),
                             padding='same'))
    fashion_model.add(MaxPooling2D((2,1),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout(0.2))

    fashion_model.add(Conv2D(16, kernel_size=(5,1),
                             activation='relu',
                             padding='same'))
    fashion_model.add(MaxPooling2D((2,1),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout(0.2))

    fashion_model.add(Conv2D(8, kernel_size=(5,1),
                             activation='relu',
                             padding='same'))
    fashion_model.add(MaxPooling2D((2,1),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout(0.2))

    fashion_model.add(Flatten())

    fashion_model.add(Dense(num_classes))

    return fashion_model
