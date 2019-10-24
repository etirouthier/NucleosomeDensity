#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:50:33 2019

@author: routhier
"""

from keras.models import Model
from keras.layers import Dropout, Concatenate, BatchNormalization
from keras.layers import Conv2D, Reshape, Input, MaxPooling2D
import keras.backend as K


def cnn_dilated_model(num_classes=1) :
    """
        Create a convolutional model with 2 convolutional layers with maxpooling before applying 
        a dilated convolutional layer to the model. The result is given by a layer with a convolution
        with size (1x1).
        
        ..notes: the precision of the prediction does not depend strongly on the architecture.
    """
    WINDOW = 2**14

    inputs = Input(shape=(WINDOW, 1, 4))
    dropout_layer = Dropout(0.1)

    image = Conv2D(128, kernel_size=(20, 4),
                          activation='relu',
                          padding='same')(inputs)
    image = BatchNormalization()(image)
    image = MaxPooling2D((2,1),padding='same')(image)
    image = dropout_layer(image)

    image = Conv2D(128, kernel_size=(7, 1),
                          activation='relu',
                          padding='same')(image)
    image = BatchNormalization()(image)
    image = MaxPooling2D((4,1),padding='same')(image)
    image = dropout_layer(image)

    image = Conv2D(256, kernel_size=(3,1),
                          activation='relu',
                          padding='same')(image)
    image = BatchNormalization()(image)
    image = MaxPooling2D((2,1),padding='same')(image)
    new_image = dropout_layer(image)

    for i in range(1,5):
        dilated_conv_layer = Conv2D(32, kernel_size=(3,1),
                                    activation='relu',
                                    padding='same',
                                    dilation_rate=(2**i,1))
        image_1 = dilated_conv_layer(new_image)
        image_1 = BatchNormalization()(image_1)
        image_1 = dropout_layer(image_1)
        new_image = Concatenate(axis=-1)([new_image, image_1])

    output = Conv2D(1, kernel_size=(1, 1),
                          activation='relu',
                          padding='same')(new_image)
    
    output = Reshape((K.int_shape(output)[1], 1))(output)

    return Model(inputs, output), K.int_shape(output)[1]
