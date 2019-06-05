#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:02:36 2019

@author: routhier
"""

import argparse 
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from keras.models import load_model
from scipy.stats import pearsonr


from MyModuleLibrary.array_modifier import rolling_window
from MyModuleLibrary.mykeras.losses import correlate, mse_var, mae_cor
from CustomModel.Models import model_dictionary
from DataPipeline.generator import nuc_occupancy

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight_file',
                        help = '''File containing the trained model with which
                        the prediction will be made.''')
    parser.add_argument('-d', '--directory',
                        help = '''Directory containing the DNA sequence
                        chromosome by chromosome in .hdf5
                        (in seq_chr_sacCer3)''')
    parser.add_argument('-f', '--file',
                        help= """CSV file containing the nucleosome occupancy
                        on the whole genome.""")
    parser.add_argument('-s', '--seq2seq',
                        action='store_true',
                        help='If the model is a seq2seq model')
    parser.add_argument('-m','--model',
                        help='''Name of the model to predict
                        (only is seq2seq model)''')
    return parser.parse_args(args)

def load_data(seq2seq=False, args=None):
    window = 2001
    half_wx = window // 2
    args = parse_arguments(args)
    path_to_directory = os.path.dirname(os.path.dirname(args.directory)) 
    # we get the path conducting to seq_chr_sacCer3
    path_to_file = os.path.join(path_to_directory,
                                'seq_chr_sacCer3',
                                args.directory,
                                'chr16.hdf5')

    f = h5py.File(path_to_file,'r')
    nucleotid = np.array(f[f.keys()[0]])
    f.close()

    X_one_hot = (np.arange(nucleotid.max()) == nucleotid[...,None]-1).astype(int)
    X_ = X_one_hot.reshape(X_one_hot.shape[0],
                           X_one_hot.shape[1] * X_one_hot.shape[2])

    proba_directory = os.path.dirname(args.file)
    proba_file = os.path.join(proba_directory, 'Start_data', args.file)

    proba = pd.read_csv(proba_file)
    y_true = proba[proba.chr == 'chr16'].value.values
    threshold = nuc_occupancy(proba_file, return_threshold=True)

    if seq2seq:
        _, output_len = model_dictionary()[args.model]

        if output_len % 2 == 0:
            half_len = output_len//2
        else:
            half_len = output_len//2 + 1

        X_slide = rolling_window(X_, window=(window,4), asteps=(output_len,4))
        X_ = X_slide.reshape(X_slide.shape[0],
                             X_slide.shape[2],
                             X_slide.shape[3],
                             1)
        y_true = y_true[half_wx - half_len : X_slide.shape[0]*output_len + half_wx - half_len]

    else:
        X_slide = rolling_window(X_, window=(window,4))
        X_ = X_slide.reshape(X_slide.shape[0],
                             X_slide.shape[2],
                             X_slide.shape[3],
                             1)
        y_true = y_true[half_wx : -half_wx]

    y_true /= np.max(y_true) * 1.5
    y_true /= float(threshold)

    return X_, y_true

def main(command_line_arguments=None):
    arguments = parse_arguments(args=command_line_arguments)
    results_path = os.path.join(os.path.dirname(__file__),
                                '../Results_nucleosome')

    path_to_weight = os.path.join(results_path,
                                  os.path.basename(os.path.dirname(arguments.weight_file)),
                                  os.path.basename(arguments.weight_file))
    path_to_results = os.path.join(results_path,
                                   os.path.basename(os.path.dirname(arguments.weight_file)),
                                   'y_pred' + os.path.basename(arguments.weight_file)[7 : -5])

    model = load_model(path_to_weight,
                       custom_objects={'correlate': correlate,
                                       'mse_var': mse_var,
                                       'mae_cor': mae_cor})
    X_test, y_true = load_data(arguments.seq2seq, command_line_arguments)

    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape((y_pred.shape[0]*y_pred.shape[1],))
    np.save(path_to_results, y_pred)

    correlation = pearsonr(y_pred, y_true)[0]
    print('Correlation between true and pred :', correlation)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(2,1,1)
    ax.plot(y_pred, 'b', label='prediction')
    ax.plot(y_true, 'r', label='experimental')
    ax.legend()
    ax2 = fig.add_subplot(2,1,2)
    ax2.hist(y_pred, bins=100, density=True,
             histtype='step', color='b', label='prediction')
    ax2.hist(y_true, bins=100, density=True,
             histtype='step', color='r', label='experimental')
    ax2.legend()
    ax.set_title('Experimental and predicted occupancy' + \
                 'on chr 16 for model{}'.format(arguments.weight_file[6:]))
    ax2.set_title('Experimental and predicted distribution of score' + \
                  'on chr 16 for model{}'.format(arguments.weight_file[6:]))
    #plt.show()

if __name__ == '__main__':
    main()
