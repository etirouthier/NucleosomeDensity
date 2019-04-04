#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:21:45 2019

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
from DataPipeline.generator import nuc_occupancy

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight_file',
                        help = '''File containing the trained model with which
                        the prediction will be made.''')
    parser.add_argument('-d', '--directory',
                        help='''Directory containing the DNA sequence chromosome
                        by chromosome in .hdf5 (in seq_chr_sacCer3)''')
    parser.add_argument('-n', '--nuc',
                        help="""CSV file containing the nucleosome occupancy
                        on the whole genome.""")
    parser.add_argument('-r', '--rna_seq',
                        help="""CSV file with the RNA seq landscape""")
    return parser.parse_args()

def load_data():
    args = parse_arguments()

    window_nuc = 2001
    half_wx = window_nuc // 2
    window_rna = 250
    half_wx_rna = window_rna // 2
    args = parse_arguments()
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

    nuc_directory = os.path.dirname(args.nuc)
    nuc_file = os.path.join(nuc_directory, 'Start_data', args.nuc)

    nuc_density = pd.read_csv(nuc_file)
    y_true = nuc_density[nuc_density.chr == 'chr16'].value.values

    X_slide = rolling_window(X_, window=(window_nuc,4))
    X_ = X_slide.reshape(X_slide.shape[0],
                         X_slide.shape[2],
                         X_slide.shape[3],
                         1)
    
    threshold = nuc_occupancy(nuc_file, return_threshold=True)
    
    rna_directory = os.path.dirname(args.rna_seq)
    rna_file = os.path.join(rna_directory, 'Start_data', args.rna_seq)
    
    rna_density = pd.read_csv(rna_file)
    rna_density = rna_density[rna_density.chr == 'chr16'].value.values

    rna_density[rna_density > 0] = np.log(rna_density[rna_density > 0])
    rna_density[rna_density < 0] = - np.log( -rna_density[rna_density < 0])

    rna_inputs = rolling_window(rna_density, window=(window_rna,))
    rna_inputs = rna_inputs[half_wx - half_wx_rna + 1 : -half_wx + half_wx_rna]
    rna_inputs  = rna_inputs.reshape(rna_inputs.shape[0], window_rna, 1)
    
    y_true = y_true[half_wx : -half_wx]
    y_true /= float(threshold)

    return X_, rna_inputs, y_true

def main():
    args = parse_arguments()
    results_path = '/users/invites/routhier/Documents/' + \
                   'Projet_nucleosomes/Results_nucleosome'
    path_to_weight = os.path.join(results_path, args.weight_file)
    path_to_results = os.path.join(results_path,
                                   os.path.dirname(args.weight_file),
                                   'y_pred' + os.path.basename(args.weight_file)[6 : -5])

    model = load_model(path_to_weight,
                       custom_objects={'correlate': correlate,
                                       'mse_var': mse_var,
                                       'mae_cor': mae_cor})
    X_test, rna_test, y_true = load_data()

    y_pred = model.predict([X_test, rna_test])
    y_pred = y_pred.reshape((y_pred.shape[0]*y_pred.shape[1],))
    np.save(path_to_results, y_pred)    
    
    correlation = pearsonr(y_pred, y_true)[0]
    print('Correlation between true and pred :', correlation)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(2,1,1)
    ax.plot(y_pred, 'b', label='prediction')
    ax.plot(y_true, 'r', label='experimental')
    ax.legend()
    ax2 = ax = fig.add_subplot(2,1,2)
    ax2.hist(y_pred, bins=100, density=True,
             histtype='step', color='b', label='prediction')
    ax2.hist(y_true, bins=100, density=True,
             histtype='step', color='r', label='experimental')
    ax2.legend()
    plt.title('Experimental and predicted occupancy' + \
              'on chr 16 for model{}'.format(args.weight_file[6:]))
    plt.show()

if __name__ == '__main__':
    main()