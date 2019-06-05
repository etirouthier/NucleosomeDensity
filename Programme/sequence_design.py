#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:06:09 2019

@author: routhier
"""
import numpy as np
import random
from math import exp
import matplotlib.pyplot as plt
import os
import h5py
import argparse


from keras.models import load_model


from MyModuleLibrary.array_modifier import rolling_window
from MyModuleLibrary.mykeras.losses import correlate, mse_var, mae_cor


class Sequence:


    def __init__(self, dna_seq):
        """
            Initiate a DNA sequence that can evolve throw the Metropolis 
            algorithm.
            
            Args:
                dna_seq: numpy array with the initial DNA sequence (with 1
                standing for A, 2 for T, 3 for G and 4 for C)
        """
        self.sequence = np.copy(dna_seq)
        self.length = len(dna_seq)
        self.mutated_seq = np.copy(dna_seq)
        Sequence.WX = 2001

    @property
    def seq_predictable(self):
        """
            Change the sequence into a serie of one-hot-encoded window to serve
            as input of a model.
        """
        x_seq = self._process(self.sequence)
        return x_seq

    @property
    def mutated_seq_predictable(self):
        """
            Change one nucleotid in the mutated_sequence and convert it into a
            predictable form.
        """   
        self._propose_mutation()
        x_mutated_seq = self._process(self.mutated_seq)
        return x_mutated_seq

    def accept_mutation(self):
        """If the mutation is accepted, change definitly the sequence""" 
        self.sequence = self.mutated_seq

    def reject_mutation(self):
        """If the mutation is rejected, the sequence is not changed"""
        self.mutated_seq = self.sequence

    def _one_hot_encoder(self, nucleotid):
        res = (np.arange(nucleotid.max()) == nucleotid[..., None]-1).astype(int)
        res = res.reshape(res.shape[0], 4)
        return res

    def _process(self, nucleotid):
        x = self._one_hot_encoder(nucleotid)
        x_slide = rolling_window(x, window=(Sequence.WX, 4))
        x_seq = x_slide.reshape(x_slide.shape[0], Sequence.WX, 4, 1)
        return x_seq

    def _propose_mutation(self):
        HALF_WX = Sequence.WX // 2
        position = random.randint(HALF_WX, HALF_WX + 167)
        mutation = np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.2, 0.2])

        while mutation == self.mutated_seq[position]:
            mutation = np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.2, 0.2])
        positions = [position + i*167 for i in range(5)]
        self.mutated_seq[positions] = mutation


class Energie:


    def __init__(self, seq, model, y_target):
        """
            This class is aimed at storing the previous energie so that to be
            able to reject a mutation.
            
            To initialise the energie we need the model on wich the prediction
            will be made, the target function and also the first sequence (in a
            predictable shape). For representation purpose we also keep the
            predicted nucleosome density.
            
            Args:
                seq: numpy array of the one-hot-encoded initial sequence
                (with a rolling window applied so that prediction can be made).
                model: the trained keras model.
                y_target: the target nucleosome density.
            Returns:
                energie: an Energie instance.
        """
        self.model = model
        self.y_target = y_target
        self.prediction = self.model.predict(seq)
        self.mutated_pred = np.copy(self.prediction)
        self.energie = self._get_energie(self.prediction)
        self.mutated_energie = self._get_energie(self.mutated_pred)

    def compute_mutated_energie(self, mutated_seq):
        """
            Calculate the energie of a mutated sequence and returns it.
            
            Args:
                mutated_seq: the one-hot-encoded proposed mutated sequence.
            Returns:
                Change self.mutated_energie to the new value.
        """
        self.mutated_pred = self.model.predict(mutated_seq)
        self.mutated_energie = self._get_energie(self.mutated_pred)

    def accept_mutation(self):
        self.energie = self.mutated_energie
        self.prediction = self.mutated_pred

    def reject_mutation(self):
        self.mutated_energie = self.energie
        self.mutated_pred = self.prediction

    def _get_energie(self, y):
        return np.mean(np.abs(y[:, 0] - self.y_target)) - \
               np.corrcoef(y[:, 0], self.y_target)[0, 1] + 1


def step(sequence, energie, temp):
    """
        Propose a mutation of the sequence and accept or reject this mutation
        on the basis of Metropolis algorithm.
        
        We propose a mutation and calculate the predicted density of nucleosome
        on the mutated sequence. The energie is the euclidian distance between
        the density predict and the density that we finally want to have.
        
        Args:
            sequence: a Sequence instance
            energie: an energie instance corresponding to the sequence, the
            model and the target
            temp: the temperature of the simulation
        Return:
            accepted: boolean, true if mutation accepted
    """
    energie.compute_mutated_energie(sequence.mutated_seq_predictable)
    rand = random.uniform(0,1)

    if exp((energie.energie - energie.mutated_energie) / temp) >= rand :
        sequence.accept_mutation()
        energie.accept_mutation()
        accepted = True

    else:
        sequence.reject_mutation()
        energie.reject_mutation()
        accepted = False

    return accepted

def _parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--chromosome',
                        help='''The hdf5 file with the chromosome sequence on
                        which we want to take the initial sequence''')
    parser.add_argument('--start',
                        help="""The position of the first nucleotid in the
                        initial sequence (it will insert an artificial seq of
                        167bp long in the middle)""")
    parser.add_argument('-s','--steps',
                        help='''Number of Metropolis iterations.''')
    parser.add_argument('-m','--model',
                        help='''Name of the model to be used for prediction''')
    parser.add_argument('-t', '--temperature',
                        help="""temperature used in the metropolis algorithm""")
    return parser.parse_args(args)

def main(command_line_arguments=None):
    """
        Metropolis algorithm to design a sequence that leads to a desired
        nucleosome density.
    """
    HALF_WX = 1000
    args = _parse_arguments(command_line_arguments)
    path_to_dna = os.path.join(os.path.dirname(__file__),
                               'seq_chr_sacCer3',
                               os.path.basename(os.path.dirname(args.chromosome)),
                               os.path.basename(args.chromosome))

    f = h5py.File(path_to_dna, 'r')
    nucleotid = np.array(f[f.keys()[0]])
    f.close()
    
    dna_seq = nucleotid[int(args.start) : int(args.start) + HALF_WX]
    artificial_seq = np.random.randint(1, 5, (167, 1))
    repeat = 5
    for i in range(repeat):
        dna_seq = np.append(dna_seq, artificial_seq)
    dna_seq = np.append(dna_seq,
                        nucleotid[int(args.start) + HALF_WX : \
                                  int(args.start) + 2*HALF_WX])

    model = load_model(os.path.join(os.path.dirname(__file__),
                                    '..',
                                    'Results_nucleosome',
                                    os.path.basename(os.path.abspath(args.model))),
                       custom_objects={'correlate': correlate,
                                       'mse_var': mse_var,
                                       'mae_cor': mae_cor})

    y_target_ = np.arange(0, 0.4, 0.4 / 73)
    y_target_ = np.append(y_target_, 0.4 - y_target_)
    y_target_ = np.append(y_target_, np.zeros((21,)))
    y_target = y_target_
    for i in range(repeat - 1):
        y_target = np.append(y_target, y_target_)

    sequence = Sequence(dna_seq)
    energie = Energie(sequence.seq_predictable, model, y_target)
    store_nrj = list()
    compteur = 0

    plt.ion()
    fig = plt.figure()
    ax_energie = fig.add_subplot(121)
    ax_pred = fig.add_subplot(122)

    line_energie, = ax_energie.plot(range(compteur), store_nrj)
    line_pred, = ax_pred.plot(y_target, 'r')
    line_pred, = ax_pred.plot(energie.prediction, 'b')

    for i in range(int(args.steps)):
        accepted = step(sequence, energie, float(args.temperature))
        store_nrj.append(energie.energie)

        if accepted:
            compteur += 1

        # saving the sequence with the minimal energie
        if i > 1 and energie.energie < min(store_nrj[:-1]):
            np.save(os.path.join(os.path.dirname(__file__),
                                 '..',
                                 'Results_nucleosome',
                                 'designed_sequence.npy'), sequence.sequence)

        if i % 10 == 0:

            line_energie.set_ydata(store_nrj)
            line_energie.set_xdata(range(i + 1))
            ax_energie.set_ylim(-0.01, max(store_nrj) + 0.05)
            ax_energie.set_xlim(0, 1.8*i)
            
            line_pred.set_ydata(energie.prediction)
            ax_pred.set_ylim(min(np.min(energie.prediction),
                                 np.min(y_target)) - 0.05,
                             max(np.max(energie.prediction),
                                     np.max(y_target)) + 0.05)

            fig.canvas.draw()
            plt.pause(1e-17)
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
