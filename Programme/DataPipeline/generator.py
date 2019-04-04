#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 09:34:19 2019

@author: routhier
"""

import numpy as np
import h5py
import os
import pandas as pd
from MyModuleLibrary.array_modifier import reorganize_random_multi_array

def nucleotid_arrays(path_to_directory):
    """
       Creates two arrays containing the DNA sequence in both train and 
       validation set.
       
       A directory containing the DNA sequence of all chromosome in .hdf5 
       format is needed and pass as argument. The train set is made of the
       DNA sequence coming from the chromosome 2 to 13 and the validation set 
       from chromosome 14 and 15.

        :param path_to_directory: the path to the directory
        :type path_to_directory: os path

        :Example:

        >>> train, val = nucleotid_arrays(chr_dir)
        >>> print(train[:5])
        array([1,4,2,3,3])

        ..warning:: created for the 16 chromosome of S.cerevisiae
        ..notes:: train set and validation set are respectively (chr 2 to 
        chr 13) and (chr 14, chr 15)
    """
    train_chr = range(2,14)
    val_chr = range(14,16)
    
    for i in train_chr: 
        path_to_file = os.path.join(path_to_directory, 'chr' + str(i) + '.hdf5')
        
        f = h5py.File(path_to_file,'r')
        nucleotid_ = np.array(f[f.keys()[0]])
        f.close()
    
        if (i == train_chr[0]):
            nucleotid_train = nucleotid_
        else :
            nucleotid_train = np.append(nucleotid_train, nucleotid_)
        
    for i in val_chr: 
        path_to_file = os.path.join(path_to_directory, 'chr' + str(i) + '.hdf5')
        
        f = h5py.File(path_to_file,'r')
        nucleotid_ = np.array(f[f.keys()[0]])
        f.close()
    
        if (i == val_chr[0]):
            nucleotid_val = nucleotid_
        else :
            nucleotid_val = np.append(nucleotid_val, nucleotid_)
        
    return nucleotid_train, nucleotid_val
    
def nuc_occupancy(path_to_file) :
    """
       Creates two arrays containing the nucleosome occupancy in both train 
       and validation set.
       
       A csv file containing the nucleosome occupancy for all the genome is
       needed. It should be made of 3 columns specifying the chromosome, 
       the position of the nucleotide on this chromosome and the nucleosome 
       occupancy at this position (chr, pos, value)

        :param path_to_file: the path to the .csv file
        :type path_to_file: os path

        :Example:

        >>> train, val = nuc_occupancy(nuc_occ.csv)
        >>> print(pd.read_csv(nuc_occ.csv))
        chr      pos    value
        chr1     0      0.1
        chr1     1      0.5
        ...     ...     ...
        chr16  948471   0.03
        chr16  948472   0.0
        >>> print(train)
        array([0.1, 0.5, ... ,0.2, 0.1 ])

        ..warning:: need to cut the distribution in high values.
        ..notes:: train set and validation set are respectively (chr 2 to 
        chr 13) and (chr 14, chr 15).
    """
    train_chr = range(2,14)
    val_chr = range(14,16)
    WINDOW = 1000
    
    proba = pd.read_csv(path_to_file ,sep = ',')
    proba_train = np.array(proba[proba.chr == 'chr' + str(train_chr[0])].value)

    for i in train_chr[1:] :
        proba_ = np.array(proba[proba.chr == 'chr' + str(i)].value)
        proba_train = np.append(proba_train, proba_)

    # renormalization of the data between 0 and 1
    for i in range(proba_train.shape[0]//WINDOW):
        if np.max(proba_train[i*WINDOW : (i+1)*WINDOW]) != 0: 
            proba_train[i*WINDOW : (i+1)*WINDOW] /= np.max(proba_train[i*WINDOW : (i+1)*WINDOW])

    proba_val = np.array(proba[proba.chr == 'chr' + str(val_chr[0])].value)

    for i in val_chr[1:] :
        proba_ = np.array(proba[proba.chr == 'chr' + str(i)].value)
        proba_val = np.append(proba_val, proba_)

    # renormalisation of the data between 0 and 1
    for i in range(proba_val.shape[0]//WINDOW):
        if np.max(proba_val[i*WINDOW : (i+1)*WINDOW]) != 0: 
            proba_val[i*WINDOW : (i+1)*WINDOW] /= np.max(proba_val[i*WINDOW : (i+1)*WINDOW])

    proba = np.append(proba_train, proba_val)
    bins = int(np.max(proba))
    digitize = np.digitize(proba, bins=np.linspace(0, bins, 100))

    weights_train = np.ones(proba_train.shape) 
    digitize_train = np.digitize(proba_train, bins=np.linspace(0, bins, 100))
    # the weight on the validation set are ones 
    weights_val = np.ones(proba_val.shape)
    
    unique = np.unique(digitize, return_counts=True)
    
    for index, count in zip(unique[0], unique[1])[1:]:
        weights_train[digitize_train == index] = np.max(unique[1][1:])/(float(count))

    return proba_train, weights_train, proba_val, weights_val
        
def generator(path_to_directory, path_to_file, output_len=1,
              include_zeros = False, seq2seq = False):
    """
        Creates two keras data generator for the train set and the validation 
        set.
    
        :param path_to_directory: the path to the a directory containing the 
        DNA sequence of all chromosomes in .hdf5 format (see nucleotid_arrays())
        :param path_to_file : the path to the .csv file with the nucleosome 
        occupancy (see nuc_occupancy())
        :param include_zeros: weither or not to include zeros in the traning 
        :param seq2seq: weither the label is a sequence of length window or a single value
        :param output_len: the length of the output with a seq2seq model 
        :type path_to_directory: os path
        :type path_to_file: os path
        :type include_zeros: Boolean, default = False
        :type seq2seq: Boolean, default = False
        :type output_len: integer
        
        :Example:
    
        >>> generator_train, number_of_set_train, generator_val, number_of_set_val = generator(chr_dir, nuc_occ.csv)
        >>> keras.fit_generator(generator = generator_train, 
                                steps_per_epochs = number_of_set_train, 
                                validation_data = generator_val, 
                                validation_steps = number_of_set_val)
    
        ..notes:: batch_size = 512, window = 2001 bp
        ..warning:: seq2seq need to be manually adapted to the model used
    """
    nucleotid_train, nucleotid_val = nucleotid_arrays(path_to_directory)
    proba_train, weights_train, proba_val, weights_val = nuc_occupancy(path_to_file)
    
    positions_train = np.arange(0, nucleotid_train.shape[0])
    positions_val = np.arange(0, nucleotid_val.shape[0])

    batch_size = 512
    number_of_set_train = positions_train.shape[0] // batch_size
    number_of_set_val = positions_val.shape[0] // batch_size
    
    if not include_zeros :
        positions_train = positions_train[proba_train > 0]
        positions_val = positions_val[proba_val > 0 ]
        
    positions_train = positions_train[1500 : - 1501]
    positions_val = positions_val[1500 : - 1501]
    
    def generator_function(positions, nucleotid, proba, weights) :
        window = 2001   
        number_of_set = positions.shape[0] // batch_size
        half_wx = int((window-1)/2.)
        length = int(positions.shape[0] // number_of_set)
        half_len = output_len // 2
        
        while True:
    
            # reshuffled the train set after an epoch
            position = reorganize_random_multi_array(positions)
    
            for num in range(0, number_of_set) :
                if window % 2 == 0 :
                    raise ValueError("window must be an odd number")
    
                positions_ = position[num*length : (num + 1) * length]
                X_ = np.zeros((positions_.shape[0],window ,4,1))
    
                for i in range(0, positions_.shape[0]) :
                    nucleotid_ = nucleotid[positions_[i] - half_wx : positions_[i] + half_wx + 1]
                    nucleotid_ = nucleotid_.reshape(nucleotid_.shape[0],1)
                    X_one_hot = (np.arange(4) == nucleotid_[...,None]-1).astype(int)
                    _X_ = X_one_hot.reshape(X_one_hot.shape[0], X_one_hot.shape[1] * X_one_hot.shape[2],1)
                    X_[i] = _X_
                
                if seq2seq and (output_len % 2 == 1):
                    y = np.array([proba[pos - half_len : pos + half_len + 1] for pos in positions_])
                    w = np.array([weights[pos - half_len : pos + half_len + 1] for pos in positions_])
                    y = y.reshape((y.shape[0], y.shape[1], 1))
                elif seq2seq and (output_len % 2 == 0):
                    y = np.array([proba[pos - half_len : pos + half_len] for pos in positions_])
                    w = np.array([weights[pos - half_len : pos + half_len] for pos in positions_])
                    y = y.reshape((y.shape[0], y.shape[1], 1))
                else:
                    y = proba[positions_] 
                    w = weights[positions_]
                    y = y.reshape(y.shape[0],1)
                    import pdb; pdb.set_trace()
                yield X_, y, w
                
    return generator_function(positions_train, nucleotid_train, proba_train, weights_train), number_of_set_train, generator_function(positions_val, nucleotid_val, proba_val, weights_val), number_of_set_val



























