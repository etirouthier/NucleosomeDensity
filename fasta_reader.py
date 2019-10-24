#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:14:03 2019

@author: routhier
"""

import gzip
import tempfile
import re
import argparse
import os

import numpy as np
import h5py
from Bio import SeqIO

def _converter():
    """
        Dictionnary used to convert DNA sequence into number.
    """
    dconv = {}
    dconv["N"] = 0
    dconv["n"] = 0
    dconv["A"] = 1
    dconv["a"] = 1
    dconv["T"] = 2
    dconv["t"] = 2
    dconv["G"] = 3
    dconv["g"] = 3
    dconv["C"] = 4
    dconv["c"] = 4
    dconv["K"] = 0
    dconv["f"] = 0
    dconv["M"] = 0
    dconv["R"] = 0
    dconv["Y"] = 0
    dconv["S"] = 0
    dconv["W"] = 0
    dconv["B"] = 0
    dconv["V"] = 0
    dconv["H"] = 0
    dconv["D"] = 0
    dconv["X"] = 0
    return dconv

def _convertchar(achar):
    """
        Convert one character.
    """
    dconv = _converter()
    return dconv[achar]

def _convertaline(aseq):
    """
        Convert a sequence.
    """
    i = 0
    L = len(aseq)
    aseqL = np.zeros((L, 1))

    while i < L:
        if i % 1000000 == 0:
            print('Already {} nucleotides have been converted'.format(i))
        aseqL[i] = _convertchar(aseq[i])
        i += 1
    return aseqL, L

def faconverter(filenamein, pathtofileout=False):
    '''
        Takes a .fa file converts it into an .hdf5 file. If path is pro-
        vided the hdf5 will be saved, if not the function returns a hdf5 python
        file object.

        filenamin: the .fa file that need to be converted (or a .fa.gz file),
        one chromosome only.
        pathtofileout: if passed, path to the returning hdf5 file.
    '''
    if re.match(r'.*\.fa$', os.path.basename(filenamein)):
        fin = open(filenamein, 'rt')
    elif re.match(r'.*\.fa\.gz$', os.path.basename(filenamein)):
        fin = gzip.open(filenamein, 'rt')
    else:
        raise ValueError("file must be a fasta file (or .fa.gz)")
        
    print('Converting the file : {}'.format(filenamein))
    
    for seq_record in SeqIO.parse(fin, 'fasta'):
        vout = _convertaline(seq_record.seq)[0]

    if not pathtofileout:
        tf = tempfile.TemporaryFile()
        fh5 = h5py.File(tf)
        fh5['data'] = vout
        return fh5

    else:
        fh5 = h5py.File(pathtofileout)
        fh5['data'] = vout
        fh5.close()

    fin.close()

def _parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--directory',
                        help='''Directory containing the DNA sequence
                                  chromosome by chromosome in .fa.gz
                               ''')
    parser.add_argument('-o',
                        '--output',
                        help='''global path to output directory (local)''')

    return parser.parse_args(args)

def main(command_line_arguments=None):
    """
        Converts the .fa sequence file in a directory to .hdf5 file
        (all the file chr*.fa in the directory). The output directory need to
        be parsed.
    """
    args = _parse_arguments(command_line_arguments)

    for element in os.listdir(args.directory):
        if re.match(r'chr\d+.?\.fa', element):
            num = re.search('chr\d+.?\.', element)
            faconverter(os.path.join(args.directory, element),
                        os.path.join(args.output, num.group(0) + 'hdf5'))

if __name__ == '__main__':
    main()
