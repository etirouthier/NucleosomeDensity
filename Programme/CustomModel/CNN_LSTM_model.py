#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:25:16 2019

@author: routhier
"""

from keras.models import Model
from keras.layers import Dropout, TimeDistributed, Input, BatchNormalization
from keras.layers import Dense, Conv2D, MaxPooling2D, LSTM, Reshape
import keras.backend as K

def cnn_lstm_model(num_classes=1) :
    """
        Create a convolutional model with 3 convolutional layers, an 
        LSTM layer and finally a time distributed dense layer,.
        
        ..notes: the precision of the prediction does not depend strongly with the architecture.
    """
    window = 2001

    inputs = Input(shape=(window, 1, 4))
    x = Conv2D(64, (3, 1),
               padding='same',
               activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(16,
               kernel_size=(8,1),
               activation='relu',
               padding='same')(x)
    x = MaxPooling2D((2,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(8,
               kernel_size=(80,1),
               activation='relu',
               padding='same')(x)
    x = MaxPooling2D((3, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Reshape((K.int_shape(x)[1], K.int_shape(x)[3]))(x)
    x = LSTM(20, return_sequences=False)(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs, outputs)
    return model

