#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:00:47 2019

@author: routhier
"""

from keras.models import Model, Sequential
from keras.layers import Dense, concatenate, Conv2D, MaxPooling2D, Dropout
from keras.layers import Reshape, Flatten, BatchNormalization
from keras import Input
import keras.backend as K

def cnn_model() :
    """
        Create a convolutional model with 3 convolutional layers before a final 
        dense a layer with one node used to make the final prediction.
        
        ..notes: the precision of the prediction does not depend strongly with the architecture.
    """
    window = 2001

    fashion_model = Sequential()

    fashion_model.add(Conv2D(64, kernel_size=(3,4),
                             activation='relu',
                             input_shape=(window,4,1),
                             padding='valid'))
    fashion_model.add(MaxPooling2D((2,1),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout(0.2))

    fashion_model.add(Conv2D(16, kernel_size=(8,1),
                             activation='relu',
                             padding='same'))
    fashion_model.add(MaxPooling2D((2,1),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout(0.2))

    fashion_model.add(Conv2D(8, kernel_size=(80,1),
                             activation='relu',
                             padding='same'))
    fashion_model.add(MaxPooling2D((2,1),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout(0.2))

    fashion_model.add(Flatten())
    
    return fashion_model 

def cnn_model_dna_rna():
    """
        Create a model that takes a sequence of DNA and the corresponding
        RNA seq landscape as input and returns the density of nucleosome at the
        center nucleotid of the sequence.
    """
    WINDOW = 2001
    WINDOW_RNA = 10
    
    dna_input = Input(shape=(WINDOW, 4, 1), name='dna_sequence')
    cnn_dna_model = cnn_model()
    encoded_dna = cnn_dna_model(dna_input)
    
    rna_seq_input = Input(shape=(WINDOW_RNA, 1), name='rna_seq_landscape')
    
    target_shape = (K.int_shape(rna_seq_input)[1],)
    reshaped_rna = Reshape(target_shape)(rna_seq_input)
    
    concatenated = concatenate([encoded_dna, reshaped_rna], axis=-1)
    nucleosome_density = Dense(1, activation='relu')(concatenated)

    model = Model([dna_input, rna_seq_input], nucleosome_density)

    return model
    

