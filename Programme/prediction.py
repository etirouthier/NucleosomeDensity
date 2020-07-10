#!/usr/bin/env python3
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
import keras.backend as K
import tensorflow as tf


from MyModuleLibrary.array_modifier import rolling_window
from MyModuleLibrary.mykeras.losses import correlate, mse_var, mae_cor
from CustomModel.Models import model_dictionary
from DataPipeline.generator import nuc_occupancy, _max_norm


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--weight_file",
        help="""File containing the trained model with which
                        the prediction will be made.""",
    )
    parser.add_argument(
        "-d",
        "--directory",
        help="""Directory containing the DNA sequence
                        chromosome by chromosome in .hdf5
                        (in seq_chr_sacCer3)""",
    )
    parser.add_argument(
        "-s", "--seq2seq", action="store_true", help="If the model is a seq2seq model"
    )
    parser.add_argument(
        "-m",
        "--model",
        help="""Name of the model to predict
                        (only is seq2seq model)""",
    )
    parser.add_argument(
        "--test",
        default="16",
        help="""chromosome on which to make prediction
                        (defaut 16 for S.cerevisiae)""",
    )
    parser.add_argument('-r', '--reversed_seq', action='store_true',
                        help='In order to predict the backward strand')
    return parser.parse_args(args)

def load_data(seq2seq=False, args=None):
    window = 2001
    half_wx = window // 2
    args = parse_arguments(args)
    path_to_directory = os.path.dirname(os.path.dirname(args.directory))
    # we get the path conducting to seq_chr_sacCer3
    path_to_file = os.path.join(
        path_to_directory,
        "seq_chr_sacCer3",
        args.directory,
        "chr" + args.test + ".hdf5",
    )

    f = h5py.File(path_to_file, "r")
    nucleotid = np.array(f["data"])
    f.close()

    if args.reversed_seq:
        nucleotid[nucleotid == 1] = 5
        nucleotid[nucleotid == 2] = 6
        nucleotid[nucleotid == 3] = 7
        nucleotid[nucleotid == 4] = 8
        nucleotid[nucleotid == 5] = 2
        nucleotid[nucleotid == 6] = 1
        nucleotid[nucleotid == 7] = 4
        nucleotid[nucleotid == 8] = 3

        nucleotid = nucleotid[::-1]

    X_one_hot = (np.arange(nucleotid.max()) == nucleotid[..., None] - 1).astype(int)
    X_ = X_one_hot.reshape(X_one_hot.shape[0], X_one_hot.shape[1] * X_one_hot.shape[2])

    if seq2seq:
        _, output_len = model_dictionary()[args.model]

        X_slide = rolling_window(X_, window=(window, 4), asteps=(output_len, 4))
        X_ = X_slide.reshape(X_slide.shape[0], X_slide.shape[2], 1, X_slide.shape[3])
        windows_num = len(X_slide)

    else:
        X_slide = rolling_window(X_, window=(window, 4))
        X_ = X_slide.reshape(X_slide.shape[0], X_slide.shape[2], 1, X_slide.shape[3])
        output_len = 1
        windows_num = 1

    return X_

def prepare_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  
    config.log_device_placement = True 
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess) 

def main(command_line_arguments=None):
    prepare_session()
    arguments = parse_arguments(args=command_line_arguments)
    results_path = os.path.join(os.path.dirname(__file__), "../Results_nucleosome")

    path_to_weight = os.path.join(
        results_path,
        os.path.basename(os.path.dirname(arguments.weight_file)),
        os.path.basename(arguments.weight_file),
    )
    
    if arguments.reversed_seq:
        path_to_results = os.path.join(
            results_path,
            os.path.basename(os.path.dirname(arguments.weight_file)),
            "y_pred_bw"
            + os.path.basename(arguments.weight_file)[7:-5]
            + "_applied_on_chr"
            + arguments.test,
        )
    else:
        path_to_results = os.path.join(
            results_path,
            os.path.basename(os.path.dirname(arguments.weight_file)),
            "y_pred"
            + os.path.basename(arguments.weight_file)[7:-5]
            + "_applied_on_chr"
            + arguments.test,
        )

    model = load_model(
        path_to_weight,
        custom_objects={"correlate": correlate, "mse_var": mse_var, "mae_cor": mae_cor},
    )
    X_test = load_data(arguments.seq2seq, command_line_arguments)
    y_pred = model.predict(X_test)
    np.save(path_to_results, y_pred)


if __name__ == "__main__":
    main()
