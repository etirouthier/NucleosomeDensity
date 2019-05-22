#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:00:47 2019

@author: routhier
"""

from keras.models import Model
from keras.layers import Dense, concatenate
from keras.layers import Reshape
from keras import Input
import keras.backend as K

from CustomModel.CNN_model import cnn_model

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
    
    x = Dense(32, activation='relu')(concatenated)
    nucleosome_density = Dense(1, activation='relu')(x)

    model = Model([dna_input, rna_seq_input], nucleosome_density)

    return model
    

