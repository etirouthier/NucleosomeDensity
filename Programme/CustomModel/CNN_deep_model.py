#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:17:50 2019

@author: routhier
"""

from keras.models import Model
from keras.layers import Dropout, Flatten, BatchNormalization, Add
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras import Input


def cnn_deep_model() :
    """
        Create a ResNet to predict the nucleosome density along the genome.
    """
    WINDOW = 2001
    num_classes = 1

    dna_input = Input(shape=(WINDOW, 4, 1), name='dna_sequence')

    x = Conv2D(48, kernel_size=(3,4),
                   activation='relu',
                   padding='valid')(dna_input)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, kernel_size=(3,1),
                   activation='relu',
                   padding='same')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((2,1),padding='same')(x)
    x = BatchNormalization()(x)
    
    fx = Conv2D(64, kernel_size=(3,1),
                   activation='relu',
                   padding='same')(x)
    fx = Dropout(0.2)(fx)
    fx = BatchNormalization()(fx)
    fx = Conv2D(64, kernel_size=(3,1),
                   activation='relu',
                   padding='same')(fx)
    fx = Dropout(0.2)(fx)
    x = Add()([fx, x])
    x = MaxPooling2D((2,1),padding='same')(fx)
    x = BatchNormalization()(x)
    
    fx = Conv2D(64, kernel_size=(7,1),
                   activation='relu',
                   padding='same')(x)
    fx = BatchNormalization()(fx)
    fx = Dropout(0.2)(fx)
    fx = Conv2D(64, kernel_size=(7,1),
                   activation='relu',
                   padding='same')(fx)
    fx = Dropout(0.2)(fx)
    x = Add()([fx, x])
    x = MaxPooling2D((2,1),padding='same')(fx)
    x = BatchNormalization()(x)
    
    fx = Conv2D(64, kernel_size=(12,1),
                   activation='relu',
                   padding='same')(x)
    fx = BatchNormalization()(fx)
    fx = Dropout(0.2)(fx)
    fx = Conv2D(64, kernel_size=(12,1),
                   activation='relu',
                   padding='same')(fx)
    fx = BatchNormalization()(fx)
    fx = Dropout(0.2)(fx)
    fx = Conv2D(64, kernel_size=(20,1),
                   activation='relu',
                   padding='same')(x)
    fx = BatchNormalization()(fx)
    fx = Dropout(0.2)(fx)
    x = Add()([fx, x])
    x = MaxPooling2D((2,1),padding='same')(fx)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Flatten()(x)
    x = Dense(16, activation = 'relu')(x)

    output = Dense(num_classes, activation='relu')(x)
    fashion_model = Model([dna_input], output)

    return fashion_model
