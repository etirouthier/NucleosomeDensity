#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:11:31 2019

@author: routhier
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import os

from keras.models import load_model

from MyModuleLibrary.array_modifier import rolling_window
from MyModuleLibrary.mykeras.losses import correlate, mse_var
from prediction import main as pred

def _makebatch(nuc_seq, pos, mutation, window):    
    local_seq = np.copy(nuc_seq)
    local_seq[pos] = mutation
    seq = local_seq[pos - window + 1: pos + window]
    
    x_one_hot = (np.arange(seq.max()) == seq[...,None]-1).astype(int)
    x_one_hot = x_one_hot.reshape(x_one_hot.shape[0], 4)

    x_slide = rolling_window(x_one_hot, window=(window,4))

    return x_slide.reshape(x_slide.shape[0],
                           x_slide.shape[2],
                           x_slide.shape[3],
                           1)

def _mutation_score(y_wilde, y_syn, pos, window):    
    y_true = y_wilde[pos - window + 1 : pos + 1]
    y_true = y_true.reshape(y_true.shape[0])
    
    y_syn = y_syn.reshape(y_syn.shape[0])
    return np.mean(np.abs(y_true - y_syn))    

def _parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--length',
                        help='''length of the window of the model used to 
                        make prediction''')

    return parser.parse_args(args)

def main(command_line_arguments=None):
    """
        Compute the mutation score for every single mutation possible along 
        the S.cerevisiae genome.
        
        The mutation score is the mean absolute error between the prediction of
        the model for the sequence surrounding the nucleotid with and without a
        mutation. Surronding means here being at less than a window from the 
        position.
    """
    args = _parse_arguments(command_line_arguments)
    path_to_program = os.path.dirname(__file__)
    
    model = load_model(os.path.join(path_to_program, '../Results_nucleosome/' + \
                                    'Final_Results/weights_CNN_nucleosome_' + \
                                    args.length + '_3_8_80_mse_var.hdf5'),
                       custom_objects={'correlate' : correlate,
                                       'mse_var' : mse_var})
    
    pred(['--weight_file', '/Final_Results/weights_CNN_nucleosome_' + \
                     args.length + '_3_8_80_mse_var.hdf5', '--directory',
                     'sacCer3', '--file', 'proba_normalized.csv', '--model',
                     'cnn'])

    y_pred = np.load(os.path.join(path_to_program, '../Results_nucleosome/' + \
                                    'Final_Results/y_pred_CNN_nucleosome_' + \
                                    args.length + '_3_8_80_mse_var.hdf5'))
    
    f = h5py.File(os.path.join(path_to_program,
                               '/seq_chr_sacCer3/sacCer3/chr16.hdf5'), 'r')
    nucleotid = np.array(f[f.keys()[0]])
    f.close()
    
    WINDOW = int(args.length)
    mutazome = np.zeros((nucleotid.shape[0], 4))
    
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line_mut, = ax.plot([], [])
    
    for pos in range(WINDOW, nucleotid.shape[0] - WINDOW):
        if pos % 100 == 0:
            print 'Studying nucleotid : ' + str(pos)
            
            np.save(os.path.join(path_to_program, '../Results_nucleosome/' + \
                                 'mutazome_' + args.length + '.npy'), mutazome)
            mut_score = np.sum(mutazome, axis=1)
            line_mut.set_ydata(mut_score[ : pos + 100])
            line_mut.set_xdata(np.arange(pos + 100))
            ax.set_ylim(-0.1, np.max(mut_score[ : pos]) + 0.1)
            ax.set_xlim(0, pos + 100)
            fig.canvas.draw()            

        for mutation in [1, 2, 3, 4]:
            x_syn = _makebatch(nucleotid, pos, mutation, WINDOW)
            y_syn = model.predict(x_syn)
            mutazome[pos, mutation - 1] = _mutation_score(y_pred, y_syn, pos, WINDOW)

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
         
    