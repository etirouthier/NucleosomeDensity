#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:28:40 2019

@author: routhier
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from MyModuleLibrary.array_modifier import rolling_window
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_nrl(tss_occupancy):
    """
        Calculate the NRL from the mean nucleosome occupancy in TSS region.
        
        Args:
            tss_occupancy: numpy array with the mean occupancy in TSS region
        Return:
            A scalar corresponding to the NRL
    """
    peaks = find_peaks(np.mean(tss_occupancy, axis=0), height = 0.2, distance = 130)[0]
    peaks = peaks[peaks > 480]
    nrl = np.polyfit(np.arange(1, 6, 1), peaks[:5], 1)[0]
    return nrl

def one_hot_encoder( nucleotid):
    """
        Take a nucleotid sequence and return the one-hot-encoded version.
        
        Args:
            nucleotid: array corresponding to the DNA sequence shape = (len, 1)
        returns:
            res: the array one-hot-encoded, shape=(len, 4)
    """
    res = (np.arange(nucleotid.max()) == nucleotid[..., None]-1).astype(int)
    res = res.reshape(res.shape[0], 4)
    return res

def process(nucleotid):
    """
        Take a numpy array corresponding to a DNA sequence and transform it so
        that the model is able to make prediction on it.
        
        Args:
            nucleotid: array corresponding to the DNA sequence shape = (len, 1)
        return:
            x_seq: array ready to be passed as input of a model to make
            prediction. The shape is (len, 2001, 4, 1)
    """
    WX = 2001
    x = one_hot_encoder(nucleotid)
    x_slide = rolling_window(x, window=(WX, 4))
    x_seq = x_slide.reshape(x_slide.shape[0], WX, 4, 1)
    return x_seq
    
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
    position_start = position_start[position.Stop < len(y_true) + half_wx]
    position_start = np.array(position_start)
    position_start = position_start.astype(int)

    position_stop = position.Stop[position.Start > half_wx] - half_wx
    position_stop = position_stop[position.Stop < len(y_true) + half_wx]
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

def heatmap(x, y, **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 
            # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right 

def corrplot(data, size_scale=700, marker='s',
             palette=sns.diverging_palette(20, 220, n=256),
             size_range=[0,1],
             color_range=[-1, 1]):
    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=color_range,
        palette=palette,
        size=corr['value'].abs(),
        size_range=size_range,
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale,
        )
