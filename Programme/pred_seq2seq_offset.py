#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:24:21 2019

@author: routhier
"""

import argparse 
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dropout,Flatten, LeakyReLU
from keras.layers import Dense, Conv2D, MaxPooling2D
from scipy.stats import pearsonr


from MyModuleLibrary.array_modifier import rolling_window
from MyModuleLibrary.mykeras.losses import correlate, mse_var
from CustomModel.Models import model_dictionary

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight_file', help = '''File containing the trained model with which the prediction will be made.''')
    parser.add_argument('-d', '--directory', help = '''Directory containing the DNA sequence chromosome by chromosome in .hdf5 (in seq_chr_sacCer3)''')
    parser.add_argument('-f', '--file', help= """CSV file containing the nucleosome occupancy on the whole genome.""")
    parser.add_argument('-m','--model', help='''Name of the model to predict (only is seq2seq model)''')
    return parser.parse_args()


def load_data():
    args = parse_arguments()
    
    window = 2001
    half_wx = window // 2
    args = parse_arguments()
    path_to_directory = os.path.dirname(os.path.dirname(args.directory)) # we get the path conducting to seq_chr_sacCer3
    path_to_file = os.path.join(path_to_directory, 'seq_chr_sacCer3', args.directory, 'chr16.hdf5')
  
    f = h5py.File(path_to_file,'r')
    nucleotid = np.array(f[f.keys()[0]])
    f.close()
    
    X_one_hot = (np.arange(nucleotid.max()) == nucleotid[...,None]-1).astype(int)
    X_ = X_one_hot.reshape(X_one_hot.shape[0], X_one_hot.shape[1] * X_one_hot.shape[2])
    
    proba_directory = os.path.dirname(args.file)
    proba_file = os.path.join(proba_directory, 'Start_data', args.file)
    
    proba = pd.read_csv(proba_file)
    y_true = proba[proba.chr == 'chr16'].value.values
    
    _, output_len = model_dictionary()[args.model]
    
    if output_len % 2 == 0:
        half_len = output_len//2
    else:
        half_len = output_len//2 + 1
        
    X_slide = rolling_window(X_, window=(window,4), asteps=(half_len,4))
    X_ = X_slide.reshape(X_slide.shape[0], X_slide.shape[2], X_slide.shape[3], 1)
    X_1 = X_[::2]
    X_2 = X_[1::2]
    
    y_true = y_true[half_wx : X_1.shape[0]*output_len + half_wx - half_len]

    return X_1, X_2, y_true, half_len
    
def main():
    args = parse_arguments()
    results_path = '/users/invites/routhier/Documents/Projet_nucleosomes/Results_nucleosome'
    path_to_weight = os.path.join(results_path, args.weight_file)
        
    WINDOW = 1000
    
    model = load_model(path_to_weight,custom_objects={'correlate':correlate, 'mse_var':mse_var })
    X_test_1, X_test_2, y_true , half_len= load_data()
    
    # renormalisation of the data between 0 and 1
    for i in range(y_true.shape[0]//WINDOW):
        if np.max(y_true[i*WINDOW : (i+1)*WINDOW]) != 0: 
            y_true[i*WINDOW : (i+1)*WINDOW] /= np.max(y_true[i*WINDOW : (i+1)*WINDOW])
    
    y_pred_1 = model.predict(X_test_1)
    y_pred_1 = y_pred_1.reshape((y_pred_1.shape[0]*y_pred_1.shape[1],))
    
    y_pred_2 = model.predict(X_test_2)
    y_pred_2 = y_pred_2.reshape((y_pred_2.shape[0]*y_pred_2.shape[1],))
    
    y_pred = (y_pred_1[half_len:] + y_pred_2[:-half_len])/2.
   
    correlation = pearsonr(y_pred, y_true)[0]
    print('Correlation between true and mean_pred :', correlation)
    correlation = pearsonr(y_pred_1[half_len:], y_true)[0]
    print('Correlation between true and pred_1 :', correlation)
    correlation = pearsonr(y_pred_2[:-half_len], y_true)[0]
    print('Correlation between true and pred_2 :', correlation)
    correlation = pearsonr(y_pred_2[:-half_len], y_pred_1[half_len:])[0]
    print('Correlation between pred_1 and pred_2 :', correlation)
    
    fig, ax = plt.subplots()
    ax.plot(y_pred, 'b', label = 'prediction')
    ax.plot(y_true, 'r', label = 'experimental')
    ax.legend()
    plt.title('Experimental and predicted occupancy on chr 16 for model{}'.format(args.weight_file[6:]))
    plt.show()
    
if __name__ == '__main__' :
    main()