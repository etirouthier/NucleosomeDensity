#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:28:40 2019

@author: routhier
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def position_gene(position, half_wx, y_true, ordering=True) :
    ''' 
        Position of genes, potentially ordered by size of genes if needed.
        The positions are adapted for the predicted signal.
        
        Args:
            position: pd.DataFrame with Start, Stop, Strand for every genes.
            half_wx: int, half size of the window used to predict the
            nucleosome landscape
            y_true: the experimental nucleosome landscape in the area that was
            predicted
            ordering: boolean, weither or not to order genes by their size.
        Return:
            position_start_: the corrected position (i.e taking into account
            that the prediction change the size) of the beginning of genes
            position_stop_: the end of genes (without considering their strand)
            start: positions of the TSS of genes
            stop: positions of the end of genes (in reading sense)
    '''
    position_start = position.Start[position.Start > half_wx] - half_wx
    position_start = position_start[position_start < y_true.shape[0]]
    position_start = np.array(position_start)
    position_start = position_start.astype(int)

    position_stop = position.Stop[position.Stop > half_wx + position_start[0]] - half_wx
    position_stop = position_stop[position_stop < y_true.shape[0]]
    position_stop = np.array(position_stop)
    position_stop = position_stop.astype(int)


    start = position.Start[position.Strand == '+']
    start = start[start > half_wx] - half_wx
    start = start[start < y_true.shape[0]]
    start = np.array(start)
    start = start.astype(int)

    stop = position.Stop[position.Strand == '-']
    stop = stop[stop > half_wx + position_start[0]] - half_wx
    stop = stop[stop < y_true.shape[0]]
    stop = np.array(stop)
    stop = stop.astype(int)

    if position_stop.shape[0] != position_start.shape[0] :

        position_start = np.delete(position_start, position_start.shape[0]-1,0)
    
    if ordering:
        gene_size = position_stop - position_start
        order = np.argsort(gene_size)

        position_start_ = position_start[order]
        position_stop_ = position_stop[order]
    
    else :
        position_start_ = position_start
        position_stop_ = position_stop
    
    return position_start_, position_stop_, start, stop



def NFR_position(y, half_wx, position):
    ''' 
        Finding NFR position for every genes in either predicted or real signal
        The positions are adapted for the predicted signal.
        
        NFR are defined by the first zero in the first serie of 60 or more 
        zeros in a window of 1000 bp before the TSS. The signal is set to zeros
        for every values inferior or equal to 0.4 the mean. The NFR is defined 
        as the first zeros before the TSS in the window if no such a serie 
        occures.
        
        Args:
            y: the signal on which we want to find the NFR.
            half_wx: half size of the window used to predict the nucleosome 
            landscape
            
        returns:
            nfr_position: pd.DataFrame with nfr_pos, and strand
    '''
    position_start_, position_stop_, start, stop = position_gene(position,
                                                                 half_wx,
                                                                 y)
    y /= np.mean(y)
    Y = np.copy(y.reshape((y.shape[0],)))
    wx = 1000
    Y[Y < 0.4] = 0

    for i in start :
        if (Y[i - wx : i] > 0).any():
            a = np.argwhere(Y[i - wx : i] > 0)
            a = a[1:] - a[:-1]
            a = a.reshape((a.shape[0],))
            a = a[::-1]

            if (a > 59).any() :
                b = np.argwhere(a > 59)[0][0]
                b = sum(a[:b]) + 1 
            elif (a > 1).any() :
                b = np.argwhere(a > 1)[0][0]
                b = sum(a[:b]) + 1 
            else :
                b = 0

            start[start == i] = i - b
    
    start_df = pd.DataFrame()
    start_df['nfr_pos'] = start
    start_df['strand'] = '+'
    
    for i in stop :
        if (Y[i : i + wx]>0).any():
            a = np.argwhere(Y[i : i + wx]>0)
            a = a[1:] - a[:-1]
            a = a.reshape((a.shape[0],))

            if (a > 59).any() :
                b = np.argwhere(a > 59)[0][0]
                b = sum(a[:b]) + 1 
            elif (a > 1).any() :
                b = np.argwhere(a > 1)[0][0]
                b = sum(a[:b]) + 1 
            else :
                b = 0
            
            stop[stop == i] = i + b
    
    stop_df = pd.DataFrame()
    stop_df['nfr_pos'] = stop
    stop_df['strand'] = '-'
                
    return start_df.append(stop_df)

def heat_map(y, y_true, position, half_wx, feature) :
    """
        Creates the heat-map of prediction with several possible alignments.

        The heat-map is a matrix made of several vector. A vector is the
        nucleosome occupancy in a region of 500 bp around a specific feature
        passed in input of the function. The heat-map is then all the vector of
        nucleosome occupancy around all the specific feature in the chromosome.
        Those features can be a TSS, the first nucleosome after the NFR
        detected in the experimental nucleosome occpancy or the first
        nucleosome after the NFR detected using the predicted nuc occupancy.

        Args:
            y: the nucleosome occupancy on which the heat-map will be calculated
            y_true: the experimental nucleosome occupancy
            position: pd.DataFrame with Start, Stop, Strand for every genes.
            half_wx: int, half size of the window used to predict the
            nucleosome landscape.
            feature: the feature on which the alignment will be performed.
            Takes values between 'tss', 'nuc_true', 'nuc_pred' (corresponding
            respectively to TSS, the experimental first nuc and the predicted
            one)
        Return:
            matrix: the heat-map.
    """
    X = np.copy(y.reshape((y.shape[0],)))
    X = X / np.mean(X)
    
    Y = np.copy(y_true.reshape((y_true.shape[0],)))
    Y = Y / np.mean(Y)

    length = 1000
    half_len = length // 2
    lag = 500
    height = 0.2

    if feature == 'nuc_pred':
        df_nfr = NFR_position(y, half_wx, position)
        
        nfr_plus = df_nfr[df_nfr.strand == '+'].nfr_pos.values
        nfr_minus = df_nfr[df_nfr.strand == '-'].nfr_pos.values
        
        matrix = np.array([X[nfr_plus[0] - half_len : nfr_plus[0] + length]])

        for i in nfr_plus:
            if  find_peaks(X[i : i + lag],
                           height=height,
                           distance=150)[0].any():
                offset = find_peaks(X[i : i + lag],
                                    height=height,
                                    distance=150)[0][0]
                matrix_ = np.array([X[i + offset - half_len : i + offset + length]])

                if matrix.shape[1] == matrix_.shape[1]:
                    matrix = np.append(matrix_, matrix, axis=0)
        
        for i in nfr_minus:
            if  find_peaks(X[i - lag : i ],
                           height=height,
                           distance=150)[0].any():
                offset = find_peaks(X[i - lag : i ],
                                    height = height,
                                    distance = 150)[0][-1] - lag    
                matrix_ = X[i + offset - length : i + offset + half_len]
                matrix_ = matrix_[::-1]
                matrix_ = np.array([matrix_])

                if matrix.shape[1] == matrix_.shape[1]:
                    matrix = np.append(matrix_, matrix, axis=0)
    
    elif feature == 'nuc_true':
        df_nfr = NFR_position(y_true, half_wx, position)
        
        nfr_plus = df_nfr[df_nfr.strand == '+'].nfr_pos.values
        nfr_minus = df_nfr[df_nfr.strand == '-'].nfr_pos.values
        
        matrix = np.array([X[nfr_plus[0] - half_len : nfr_plus[0] + length]])

        for i in nfr_plus:
            if  find_peaks(Y[i : i + lag],
                           height=height,
                           distance=150)[0].any():
                offset = find_peaks(Y[i : i + lag],
                                    height=height,
                                    distance=150)[0][0]
                matrix_ = np.array([X[i + offset - half_len : i + offset + length]])

                if matrix.shape[1] == matrix_.shape[1]:
                    matrix = np.append(matrix_, matrix, axis=0)
        
        for i in nfr_minus:
            if  find_peaks(Y[i - lag : i ],
                           height=height,
                           distance=150)[0].any():
                offset = find_peaks(Y[i - lag : i ],
                                    height = height,
                                    distance = 150)[0][-1] - lag    
                matrix_ = X[i + offset - length : i + offset + half_len]
                matrix_ = matrix_[::-1]
                matrix_ = np.array([matrix_])

                if matrix.shape[1] == matrix_.shape[1]:
                    matrix = np.append(matrix_, matrix, axis=0)

    elif feature == 'tss':
        position_start_, position_stop_, Start, Stop = \
        position_gene(position, half_wx, y_true, ordering=True)
        matrix = np.array([X[Start[1] - half_len : Start[1] + length]])

        for i in Start[2:] :
            matrix_ = np.array([X[i - half_len : i + length]])

            if matrix.shape[1] == matrix_.shape[1]:
                    matrix = np.append(matrix_, matrix, axis=0)

        for i in Stop :
            matrix_ = X[i - length : i + half_len]
            matrix_ = matrix_[::-1]
            matrix_ = np.array([matrix_])

            if matrix.shape[1] == matrix_.shape[1]:
                    matrix = np.append(matrix_, matrix, axis=0)

    return matrix
