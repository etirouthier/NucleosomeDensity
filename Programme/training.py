#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 09:34:19 2019

@author: routhier
"""
import os
import argparse
import re
import numpy as np

import keras.backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from MyModuleLibrary.mykeras.losses import correlate, mae_cor
from DataPipeline.generator import generator
from CustomModel.Models import model_dictionary


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory',
                        help='''Directory containing the DNA sequence chromosome
                        by chromosome in .hdf5 (in seq_chr_sacCer3)''')
    parser.add_argument('-f', '--file',  nargs='+',
                        help="""CSV file containing the nucleosome occupancy
                        on the whole genome.""")
    parser.add_argument('-o', '--output_file',
                        help="""Name of the file were the weight will be stored
                        (in Results_nucleosome)""")
    parser.add_argument('-m','--model',
                        help='''Name of the model to be trained''')
    parser.add_argument('-z', '--include_zeros', action='store_true',
                        help="""Weither or not to include zeros in the training""")
    parser.add_argument('-s', '--seq2seq', action='store_true',
                        help="""If the model is a seq2seq model""")
    parser.add_argument('-ds', '--downsampling', action='store_true',
                        help="""To downsampled the predicted sequence for a 
                        seq2seq model, the length of sampling will be calculated""")
    parser.add_argument('-n', '--norm_max', action='store_true',
                        help="""Normalizing the data by dividing by a rolling
                        max""")
    parser.add_argument('-t','--training_set', nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                        help='''list of chromosome in the training set''')
    parser.add_argument('-v','--validation_set', nargs='+',
                        default=[14, 15],
                        help='''list of chromosome in the validation set''')
    parser.add_argument('-p', '--pourcentage',
                        help='''pourcentage of the training data to be included''')
    parser.add_argument('--fft', action='store_true',
                        help="""Weither or not to apply an fft transform of the target""")
    return parser.parse_args()

def prepare_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  
    config.log_device_placement = True 
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)

def main():
    args = parse_arguments()
    prepare_session()
    path_to_directory = os.path.dirname(__file__)
    # we get the path conducting to seq_chr_sacCer3
    path_to_tensorboard = os.path.join(path_to_directory, 'Tensorboard')

    path_to_file = [os.path.join(path_to_directory,
                                'Start_data',
                                file_name) for file_name in args.file]
    path_to_directory = os.path.join(path_to_directory,
                                     'seq_chr_sacCer3',
                                     args.directory)
    num_epochs = 200
    num_classes = len(path_to_file)

    assert re.match(r'weights_.+\.hdf5', os.path.basename(args.output_file))
    path_to_output_file = os.path.join('../Results_nucleosome', args.output_file)

    if args.seq2seq :
        model, output_len = model_dictionary(num_classes)[args.model]
        generator_train, number_of_set_train, \
        generator_val, number_of_set_val = generator(path_to_directory,
                                                     path_to_file,
                                                     args.training_set,
                                                     args.validation_set,
                                                     output_len,
                                                     args.include_zeros,
                                                     args.norm_max, 
                                                     args.seq2seq,
                                                     args.downsampling,
                                                     args.pourcentage,
                                                     args.fft)
        model.compile(optimizer='adam', loss=mae_cor,
                      metrics=['mse', correlate],
                      sample_weight_mode='temporal')
    else:
        model = model_dictionary(num_classes)[args.model]
        generator_train, number_of_set_train, \
        generator_val, number_of_set_val = generator(path_to_directory,
                                                     path_to_file,
                                                     args.training_set,
                                                     args.validation_set,
                                                     1,
                                                     args.include_zeros,
                                                     args.norm_max,
                                                     args.seq2seq,
                                                     args.downsampling,
                                                     args.pourcentage,
                                                     args.fft)

        model.compile(optimizer='adam',
                      loss=mae_cor,
                      metrics=['mae', correlate])

    checkpointer = ModelCheckpoint(filepath=path_to_output_file,
                                   monitor='val_loss',
                                   verbose=0, 
                                   save_best_only=True, 
                                   save_weights_only=False, 
                                   mode='min', period=1)
    early = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=2,
                          verbose=0,
                          mode='auto')
    tensorboard = TensorBoard(log_dir=path_to_tensorboard, update_freq=200)
    print(model.summary())
    history = model.fit_generator(generator=generator_train,
                                  steps_per_epoch=number_of_set_train, #500
                                  epochs=num_epochs,
                                  validation_data=generator_val, 
                                  validation_steps=number_of_set_val, #200
                                  callbacks=[checkpointer, early, tensorboard])

if __name__ == '__main__':
    main()
