#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:11:31 2019

@author: routhier
"""

import numpy as np
import h5py
import argparse
import os

from keras.models import load_model

from MyModuleLibrary.array_modifier import rolling_window
from MyModuleLibrary.mykeras.losses import correlate, mae_cor
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

def _mutation_score(y_wild, y_syn, pos, window):    
    y_true = y_wild[pos - window + 1 : pos + 1]
    y_true = y_true.reshape(len(y_true))
    
    y_syn = y_syn.reshape(len(y_syn))
    return np.mean(np.abs(y_syn - y_true)) - \
               np.corrcoef(y_syn, y_true)[0, 1] + 1

def _parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        help = '''model with which the prediction will be made
                        ''')
    parser.add_argument('-d',
                        '--directory',
                        help = '''directory that contains the dna seq in hdf5.
                        ''')
  
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
    
    model = load_model(os.path.join(path_to_program,
                                    '..',
                                    'Results_nucleosome',
                                    os.path.basename(args.model)),
                       custom_objects={'correlate' : correlate,
                                       'mae_cor' : mae_cor})
    
    pred(['--weight_file', os.path.basename(args.model), '--directory',
          args.directory, '--file', 'proba_normalized.csv',
          '--model', 'cnn'])

    y_pred = np.load(os.path.join(path_to_program,
                                  '..',
                                  'Results_nucleosome',
                                  'y_pred' + os.path.basename(args.model)[7 : -5]) + '.npy')
    
    f = h5py.File(os.path.join(path_to_program, 'seq_chr_sacCer3',
                               args.directory, 'chr16.hdf5'), 'r')
    nucleotid = np.array(f[f.keys()[0]])
    f.close()
    
    WINDOW = 2001
    mutazome = np.zeros((len(nucleotid), y_pred.shape[1], 4))
    
    for pos in range(WINDOW, len(nucleotid) - WINDOW):
        if pos % 100 == 0:
            print('Studying nucleotid : ' + str(pos))
            
            np.save(os.path.join(path_to_program, '..','Results_nucleosome',
                                 'mutazome' + os.path.basename(args.model)[7 : -5]),
                                 mutazome)

        for mutation in [1, 2, 3, 4]:
            x_syn = _makebatch(nucleotid, pos, mutation, WINDOW)
            y_syn = model.predict(x_syn)
            
            for i in range(y_pred.shape[1]):
                mutazome[pos, mutation - 1] = _mutation_score(y_pred[:, i],
                                                              y_syn[:, i],
                                                              pos,
                                                              WINDOW)

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
         
    