#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:37:57 2019

@author: routhier
"""

import numpy as np
import pandas as pd

from MyModuleLibrary.array_modifier import reorganize_random_multi_array
from DataPipeline.generator import nucleotid_arrays, nuc_occupancy

def rna_seq_density(path_to_file):
    """
       Creates two arrays containing the RNA_seq density in both train
       and validation set and their corresponding weight arrays.

       A csv file containing the RNA_seq density for all the genome is
       needed. It should be made of 3 columns specifying the chromosome,
       the position of the nucleotide on this chromosome and the RNA_seq
       density at this position (chr, pos, value)

        :param path_to_file: the path to the .csv file
        :type path_to_file: os path

        :Example:

        >>> train, weight_train, val, weight_val = rna_seq_density(nuc_occ.csv)
        >>> print(pd.read_csv(rna_seq_dens.csv))
        chr      pos    value
        chr1     0      0.1
        chr1     1      0.5
        ...     ...     ...
        chr16  948471   0.03
        chr16  948472   0.0
        >>> print(train)
        array([0.1, 0.5, ... ,0.2, 0.1 ])

        ..warning:: need to cut the distribution in high values.
        ..notes:: train set and validation set are respectively
        (chr 2,3 and 5 to chr 14) and (chr 15, chr 16)
    """
    train_chr = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    val_chr = [14, 15]

    rna_density = pd.read_csv(path_to_file)

    rna_density_train = np.array(rna_density[rna_density.chr == 'chr' + \
                                             str(train_chr[0])].value)

    for i in train_chr[1:]:
        proba_ = np.array(rna_density[rna_density.chr == 'chr' + str(i)].value)
        rna_density_train = np.append(rna_density_train, proba_)

    # Renormalization by taking the log
    rna_density_train = rna_density_train.astype(np.float32)
    
    rna_density_train[rna_density_train > 0] = \
    np.log(rna_density_train[rna_density_train > 0])
    
    rna_density_train[rna_density_train < 0] = \
    -np.log(-rna_density_train[rna_density_train < 0])

    rna_density_val = np.array(rna_density[rna_density.chr == 'chr' + \
                                     str(val_chr[0])].value)

    for i in val_chr[1:]:
        proba_ = np.array(rna_density[rna_density.chr == 'chr' + str(i)].value)
        rna_density_val = np.append(rna_density_val, proba_)

    # Renormalization by taking the log
    rna_density_val = rna_density_val.astype(np.float32)
    
    rna_density_val[rna_density_val > 0] = \
    np.log(rna_density_val[rna_density_val > 0])
    
    rna_density_val[rna_density_val < 0] = \
    -np.log(-rna_density_val[rna_density_val < 0])
   
    return rna_density_train, rna_density_val

def generator(path_to_directory, path_to_file_rna, path_to_file_nuc,
              include_zeros= False):
    """
        Creates two keras data generator for the train set and the validation
        set with inputs DNA sequence and the corresponding RNA seq and output 
        the density of nucleosome at the center of every window.

        :param path_to_directory: the path to the a directory containing the
        DNA sequence of all chromosomes in .hdf5 format (see nucleotid_arrays)
        :param path_to_file_rna: path to the .csv file containing the rna_seq
        landscape
        :param path_to_file_nuc : the path to the .csv file with the nucleosome
        occupancy (see nuc_occupancy())
        :param include_zeros: weither or not to include zeros in the traning
        :param output_len: the length of the output with a seq2seq model
        :type path_to_directory: os path
        :type path_to_file: os path
        :type include_zeros: Boolean, default = False
        :type seq2seq: Boolean, default = False
        :type output_len: integer

        :Example:

        >>> generator_train, number_of_set_train, generator_val,\
            number_of_set_val = generator(chr_dir, nuc_occ.csv)
        >>> keras.fit_generator(generator = generator_train,
                                steps_per_epochs = number_of_set_train,
                                validation_data = generator_val,
                                validation_steps = number_of_set_val)

        ..notes:: batch_size = 512, window = 2001 bp
        ..warning:: seq2seq need to be manually adapted to the model used
    """
    nucleotid_train, nucleotid_val = nucleotid_arrays(path_to_directory)
    rna_train, rna_val = rna_seq_density(path_to_file_rna)
    proba_train, weights_train, proba_val, weights_val = \
    nuc_occupancy(path_to_file_nuc)

    positions_train = np.arange(0, nucleotid_train.shape[0])
    positions_val = np.arange(0, nucleotid_val.shape[0])
    
    if not include_zeros :
        positions_train = positions_train[proba_train > 0]
        positions_val = positions_val[proba_val > 0 ]
    
    batch_size = 512
    number_of_set_train = positions_train.shape[0] // batch_size
    number_of_set_val = positions_val.shape[0] // batch_size

    positions_train = positions_train[1500 : - 1501]
    positions_val = positions_val[1500 : - 1501]

    def generator_function(positions, nucleotid, rna, proba, weights):
        """
            The generator which will be used by the keras model to train.
        """
        window = 2001
        window_rna = 10
        number_of_set = positions.shape[0] // batch_size
        half_wx = int((window - 1) / 2.)
        half_rna = window_rna // 2
        length = int(positions.shape[0] // number_of_set)

        while True:

            # reshuffled the train set after an epoch
            position = reorganize_random_multi_array(positions)

            for num in range(0, number_of_set):
                if window % 2 == 0:
                    raise ValueError("window must be an odd number")

                positions_ = position[num*length : (num + 1) * length]
                seq = np.zeros((positions_.shape[0], window, 4, 1))
                activity = np.zeros((positions_.shape[0], window_rna, 1))

                for i in range(0, positions_.shape[0]):
                    nucleotid_ = nucleotid[positions_[i] - half_wx :
                                           positions_[i] + half_wx + 1]
                    nucleotid_ = nucleotid_.reshape(nucleotid_.shape[0], 1)
                    X_one_hot = (np.arange(4) == nucleotid_[..., None]-1).astype(int)
                    _X_ = X_one_hot.reshape(X_one_hot.shape[0],
                                            X_one_hot.shape[1] * X_one_hot.shape[2], 1)
                    seq[i] = _X_
                    
                    activity_ = rna[positions_[i] - half_rna :
                                           positions_[i] + half_rna]
                    activity_ = activity_.reshape((window_rna, 1))
                    activity[i] = activity_

                y = proba[positions_]
                w = weights[positions_]
                y = y.reshape(y.shape[0], 1)
                
                yield [seq, activity], y, w

    return generator_function(positions_train,
                              nucleotid_train,
                              rna_train,
                              proba_train,
                              weights_train), \
           number_of_set_train, \
           generator_function(positions_val,
                              nucleotid_val,
                              rna_val,
                              proba_val,
                              weights_val), \
           number_of_set_val