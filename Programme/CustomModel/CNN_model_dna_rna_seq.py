#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:00:47 2019

@author: routhier
"""

from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, BatchNormalization, MaxPooling1D
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling2D, concatenate
from keras.layers import Reshape
from keras import Input
import keras.backend as K

def _cnn_dna_model():
    cnn_model = Sequential()
    
    cnn_model.add(Conv2D(64, kernel_size=(3,4),
                             activation='relu',
                             padding='valid'))
    cnn_model.add(MaxPooling2D((2,1),padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.2))
    
    cnn_model.add(Conv2D(16, kernel_size=(8,1),
                             activation='relu',
                             padding='same'))
    cnn_model.add(MaxPooling2D((2,1),padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.2))
    
    cnn_model.add(Conv2D(8, kernel_size=(80,1),
                             activation='relu',
                             padding='same'))
    cnn_model.add(MaxPooling2D((2,1),padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.2))
    
    return cnn_model

def _cnn_rna_model():
    cnn_model = Sequential()
    
    cnn_model.add(Conv1D(8, kernel_size=3,
                             padding='valid'))
    cnn_model.add(MaxPooling1D(2, padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.2))
    
    cnn_model.add(Conv1D(16, kernel_size=3,
                             padding='same'))
    cnn_model.add(MaxPooling1D(2,padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.2))
    
    cnn_model.add(Conv1D(8, kernel_size=3,
                             padding='same'))
    cnn_model.add(MaxPooling1D(2,padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.2))
    
    return cnn_model

def cnn_model_dna_rna():
    """
        Create a model that takes a sequence of DNA and the corresponding
        RNA seq landscape as input and returns the density of nucleosome at the
        center nucleotid of the sequence.
    """
    WINDOW = 2001
    WINDOW_RNA = 250
    
    dna_input = Input(shape=(WINDOW, 4, 1), name='dna_sequence')
    cnn_dna_model = _cnn_dna_model()
    encoded_dna = cnn_dna_model(dna_input)
    
    target_shape = (K.int_shape(encoded_dna)[1], K.int_shape(encoded_dna)[-1])
    encoded_dna = Reshape(target_shape)(encoded_dna)
    
    rna_seq_input = Input(shape=(WINDOW_RNA, 1), name='rna_seq_landscape')
    cnn_rna_seq_model = Conv1D(16, kernel_size=3,
                             padding='same') #_cnn_rna_model()
    encoded_rna_seq = cnn_rna_seq_model(rna_seq_input)
    
    concatenated = concatenate([encoded_dna, encoded_rna_seq], axis=-1)
    flattened = Flatten()(concatenated)
    nucleosome_density = Dense(1, activation='relu')(flattened)
    
    model = Model([dna_input, rna_seq_input], nucleosome_density)
    
    return model
    

