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
import scipy
from MyModuleLibrary.array_modifier import reorganize_random_multi_array, rolling_window


def nucleotid_arrays(path_to_directory,
                     train_chr,
                     val_chr):
    """
       Creates two arrays containing the DNA sequence in both train and 
       validation set.
       
       A directory containing the DNA sequence of all chromosome in .hdf5 
       format is needed and pass as argument. The train set is made of the
       DNA sequence coming from the chromosome 2 to 13 and the validation set 
       from chromosome 14 and 15.

        :param path_to_directory: the path to the directory
        :type path_to_directory: os path
        :param train_chr: list of chr to be part of the training set
        :type train_chr: list
        :param val_chr: same for validation

        :Example:

        >>> train, val = nucleotid_arrays(chr_dir)
        >>> print(train[:5])
        array([1,4,2,3,3])

        ..warning:: created for the 16 chromosome of S.cerevisiae
        ..notes:: train set and validation set are respectively (chr 2 to 
        chr 13) and (chr 14, chr 15)
    """
    for i in train_chr: 
        path_to_file = os.path.join(path_to_directory, 'chr' + str(i) + '.hdf5')
        
        f = h5py.File(path_to_file,'r')
        nucleotid_ = np.array(f['data'])
        f.close()

        #nucleotid_[nucleotid_ == 2] = 5
        #nucleotid_[nucleotid_ == 4] = 2
        #nucleotid_[nucleotid_ == 5] = 4

        if (i == train_chr[0]):
            nucleotid_train = nucleotid_
        else :
            nucleotid_train = np.append(nucleotid_train, nucleotid_)

    for i in val_chr: 
        path_to_file = os.path.join(path_to_directory, 'chr' + str(i) + '.hdf5')
        
        f = h5py.File(path_to_file,'r')
        nucleotid_ = np.array(f['data'])
        f.close()
        
        #nucleotid_[nucleotid_ == 2] = 5
        #nucleotid_[nucleotid_ == 4] = 2
        #nucleotid_[nucleotid_ == 5] = 4
    
        if (i == val_chr[0]):
            nucleotid_val = nucleotid_
        else :
            nucleotid_val = np.append(nucleotid_val, nucleotid_)

    return nucleotid_train, nucleotid_val

def _find_threshold(proba):
    return np.percentile(proba, 99)
    #density, values = np.histogram(proba, bins=1000, density=True)
    #limit = 0.0001

    #if (density < limit).any():
    #    threshold = values[np.where(density < limit)[0][0]]
    #else:
    #    threshold = np.max(proba)

    #return threshold

def _max_norm(y, wx=3001):
    y_roll = rolling_window(y, window=wx)
    max_roll = np.max(y_roll, axis=1).astype(float)
    half = wx // 2
    y = np.concatenate((y[:half], y[half : - half] / max_roll, y[-half:]),
                       axis=0)
    return y

def nuc_occupancy(path_to_file,
                  train_chr,
                  val_chr,
                  max_norm=False,
                  return_threshold=False):
    """
       Creates two arrays containing the nucleosome occupancy in both train 
       and validation set.
       
       A csv file containing the nucleosome occupancy for all the genome is
       needed. It should be made of 3 columns specifying the chromosome, 
       the position of the nucleotide on this chromosome and the nucleosome 
       occupancy at this position (chr, pos, value)

        :param path_to_file: the path to the .csv file
        :type path_to_file: os path
        :param return_threshold: if true only return the threshold used for
        normalization.
        :param train_chr: list of chr to be part of the training set
        :type train_chr: list
        :param val_chr: same for validation

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
    proba = pd.read_csv(path_to_file ,sep = ',')
    proba_train = np.array(proba[proba.chr == 'chr' + str(train_chr[0])].value)
    
    for i in train_chr[1:] :
        proba_ = np.array(proba[proba.chr == 'chr' + str(i)].value)
        proba_train = np.append(proba_train, proba_)

    # renormalization of the data between 0 and 1
    threshold = _find_threshold(proba_train)
    proba_train[proba_train > threshold] =  threshold
    proba_train /= float(threshold)

    proba_val = np.array(proba[proba.chr == 'chr' + str(val_chr[0])].value)

    for i in val_chr[1:] :
        proba_ = np.array(proba[proba.chr == 'chr' + str(i)].value)
        proba_val = np.append(proba_val, proba_)

    # renormalization of the data between 0 and 1
    proba_val[proba_val > threshold] =  threshold
    proba_val /= float(threshold)
    
    if max_norm:
        proba_train = _max_norm(proba_train)
        proba_val = _max_norm(proba_val)

    proba = np.append(proba_train, proba_val)
    bins = int(np.max(proba))
    digitize = np.digitize(proba, bins=np.linspace(0, bins, 100))

    weights_train = np.ones(proba_train.shape) 
    digitize_train = np.digitize(proba_train, bins=np.linspace(0, bins, 100))
    # the weight on the validation set are ones 
    weights_val = np.ones(proba_val.shape)

    unique = np.unique(digitize, return_counts=True)

    for index, count in list(zip(unique[0], unique[1]))[1:]:
        weights_train[digitize_train == index] = np.max(unique[1][1:])/(float(count))

    if return_threshold:
        return threshold
    else:
        return proba_train.reshape(len(proba_train), 1), \
               weights_train.reshape(len(weights_train), 1), \
               proba_val.reshape(len(proba_val), 1), \
               weights_val.reshape(len(weights_val), 1)

def _calculate_rolling_mean(x, batch_size, sample_len, output_len, num_classes):
    x = rolling_window(x,
                       window=(x.shape[0], sample_len, num_classes),
                       asteps=(x.shape[0], sample_len, num_classes))
    x = x.reshape((output_len, batch_size, sample_len,num_classes))
    x = np.mean(x, axis=2)
    x = np.swapaxes(x, 0, 1)
    return x

def generator(path_to_directory,
              paths,
              train_chr,
              val_chr,
              output_len=1,
              max_norm=False,
              include_zeros=False,
              seq2seq=False,
              downsampling=False,
              pourc=None,
              fft=False):
    """
        Creates two keras data generator for the train set and the validation 
        set.
    
        :param path_to_directory: the path to the a directory containing the 
        DNA sequence of all chromosomes in .hdf5 format (see nucleotid_arrays())
        :param paths : the paths to the .csv files with the nucleosome 
        occupancy (see nuc_occupancy())
        :param include_zeros: weither or not to include zeros in the traning 
        :param seq2seq: weither the label is a sequence of length window or a 
        single value
        :param output_len: the length of the output with a seq2seq model 
        :type path_to_directory: os path
        :type paths: list of os path
        :type include_zeros: Boolean, default = False
        :type seq2seq: Boolean, default = False
        :type output_len: integer
        :param train_chr: list of chr to be part of the training set
        (defaut adapted for S.cerevisiae)
        :type train_chr: list
        :param val_chr: same for validation
        :param pourc: pourcentage of the data to be included
        :type pourc: int or None
        :param fft: applying an fft transform to the target
        :type fft: boolean
        
        :Example:
    
        >>> generator_train, number_of_set_train, generator_val, 
        number_of_set_val = generator(chr_dir, nuc_occ.csv)
        >>> keras.fit_generator(generator = generator_train, 
                                steps_per_epochs = number_of_set_train, 
                                validation_data = generator_val, 
                                validation_steps = number_of_set_val)
    
        ..notes:: batch_size = 512, window = 2001 bp
        ..warning:: seq2seq need to be manually adapted to the model used
    """
    nucleotid_train, nucleotid_val = nucleotid_arrays(path_to_directory,
                                                      train_chr,
                                                      val_chr)
    proba_train, weights_train, proba_val, weights_val = nuc_occupancy(paths[0],
                                                                       train_chr,
                                                                       val_chr,
                                                                       max_norm)
    
    for path in paths[1:]:
        proba_train_, weights_train_, proba_val_, weights_val_ = nuc_occupancy(path,
                                                                               train_chr,
                                                                               val_chr,
                                                                               max_norm)
        proba_train = np.append(proba_train, proba_train_, axis=1)
        weights_train = np.append(weights_train, weights_train_, axis=1)
        proba_val = np.append(proba_val, proba_val_, axis=1)
        weights_val = np.append(proba_train, proba_train_, axis=1)
   
    weights_train = np.mean(weights_train, axis=1)
    weights_val = np.mean(weights_val, axis=1)

    positions_train = np.arange(0, nucleotid_train.shape[0])
    positions_val = np.arange(0, nucleotid_val.shape[0]) 
    
    batch_size = 512
    
    if not include_zeros :
        positions_train = positions_train[np.mean(proba_train, axis=1) > 0]
        positions_val = positions_val[np.mean(proba_val, axis=1) > 0 ]

    positions_train = positions_train[1000 : - 1000]
    positions_val = positions_val[1000 : - 1000]

    if pourc:
        positions_train = reorganize_random_multi_array(positions_train)
        nb_examples = int(pourc) * len(positions_train) // 100
        positions_train = positions_train[:nb_examples]
 
    number_of_set_train = len(positions_train) // batch_size
    number_of_set_val = len(positions_val) // batch_size

    def generator_function(positions, nucleotid, proba, weights) :
        window = 2001   
        number_of_set = positions.shape[0] // batch_size
        half_wx = window // 2
        length = int(positions.shape[0] // number_of_set)
        half_len = output_len // 2
        num_classes = proba.shape[1]

        while True:

            # reshuffled the train set after an epoch
            position = reorganize_random_multi_array(positions)

            for num in range(0, number_of_set) :

                positions_ = position[num*length : (num + 1) * length]
                X_ = np.zeros((positions_.shape[0], window, 1, 4))
                #X_ = np.zeros((positions_.shape[0], window ,4))

                for i in range(0, positions_.shape[0]) :
                    nucleotid_ = nucleotid[positions_[i] - half_wx : positions_[i] + half_wx + (window % 2)]
                    nucleotid_ = nucleotid_.reshape(nucleotid_.shape[0],1)
                    X_one_hot = (np.arange(4) == nucleotid_[...,None]-1).astype(int)
                    _X_ = X_one_hot.reshape(X_one_hot.shape[0], 1, X_one_hot.shape[1] * X_one_hot.shape[2])
                    #_X_ = X_one_hot.reshape(X_one_hot.shape[0], X_one_hot.shape[1] * X_one_hot.shape[2])
                    X_[i] = _X_

                if seq2seq and not downsampling:
                    y = np.array([proba[pos - half_len : pos + half_len + (output_len % 2)] for pos in positions_])
                    w = np.array([weights[pos - half_len : pos + half_len + (output_len % 2)] for pos in positions_])
                    y = y.reshape((y.shape[0], y.shape[1], num_classes))
                elif seq2seq and downsampling:
                    y = np.array([proba[pos - half_wx : pos + half_wx + (window % 2)] for pos in positions_])
                    w = np.array([weights[pos - half_wx : pos + half_wx + (window % 2)] for pos in positions_])
                    y = y.reshape((y.shape[0], y.shape[1], num_classes))
                    w = w.reshape((w.shape[0], w.shape[1], 1))
                    
                    sample_len = window // output_len
                    y = _calculate_rolling_mean(y,
                                                batch_size,
                                                sample_len,
                                                output_len,
                                                num_classes)
                    w = _calculate_rolling_mean(w,
                                                batch_size,
                                                sample_len,
                                                output_len,
                                                1)
                    w = w.reshape((w.shape[0], w.shape[1]))
                else:
                    y = proba[positions_]
                    w = weights[positions_]
                    y = y.reshape(y.shape[0], num_classes)
                if fft:
                    y = np.concatenate([scipy.fft(y[:, i]).reshape((-1, 1)) for i in range(num_classes)], axis=1)
                    yield X_, y
                else:
                    yield X_, y, w
        
    return generator_function(positions_train,
                              nucleotid_train,
                              proba_train, weights_train), \
           number_of_set_train, \
           generator_function(positions_val,
                              nucleotid_val,
                              proba_val,
                              weights_val), \
           number_of_set_val
