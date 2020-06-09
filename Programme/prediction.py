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
        "-f",
        "--file",
        nargs="+",
        help="""CSV file containing the nucleosome occupancy
                        on the whole genome.""",
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
        "-t",
        "--training_set",
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        help="""list of chromosome in the training set""",
    )
    parser.add_argument(
        "-v",
        "--validation_set",
        nargs="+",
        default=[14, 15],
        help="""list of chromosome in the validation set""",
    )
    parser.add_argument(
        "--test",
        default="16",
        help="""chromosome on which to make prediction
                        (defaut 16 for S.cerevisiae)""",
    )
    parser.add_argument(
        "-n",
        "--norm_max",
        action="store_true",
        help="""Normalizing the data by dividing by a rolling
                        max""",
    )
    parser.add_argument('-r', '--reversed_seq', action='store_true',
                        help='In order to predict the backward strand')
    return parser.parse_args(args)


def _load_true(
    proba_file,
    half_wx,
    output_len=1,
    max_norm=False,
    seq2seq=False,
    windows_num=1,
    training_set=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    validation_set=[14, 15],
    chromosome="16",
):
    proba = pd.read_csv(proba_file)
    y_true = proba[proba.chr == "chr" + chromosome].value.values
    threshold = nuc_occupancy(
        proba_file, training_set, validation_set, return_threshold=True
    )

    if seq2seq:
        if output_len % 2 == 0:
            half_len = output_len // 2
        else:
            half_len = output_len // 2 + 1

        y_true = y_true[
            half_wx - half_len : windows_num * output_len + half_wx - half_len
        ]

    else:
        y_true = y_true[half_wx:-half_wx]

    y_true /= float(threshold)
    y_true[y_true > 1] = 1

    if max_norm:
        y_true = _max_norm(y_true)
    return y_true.reshape((len(y_true), 1))


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

    proba_files = [
        os.path.join(os.path.dirname(__file__), "Start_data", file_name)
        for file_name in args.file
    ]

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

    #y_true = _load_true(
    #    proba_files[0],
    #    half_wx,
    #    args.norm_max,
    #    output_len,
    #    seq2seq,
    #    windows_num,
    #    args.training_set,
    #    args.validation_set,
    #    args.test,
    #)
    #for proba_file in proba_files[1:]:
    #    y_true_ = _load_true(
    #        proba_file,
    #        half_wx,
   #         args.norm_max,
    #        output_len,
   #         seq2seq,
   #         windows_num,
     #       args.training_set,
     #       args.validation_set,
       #     args.test,
      #  )
      #  y_true = np.append(y_true, y_true_, axis=1)

    return X_#, y_true


def main(command_line_arguments=None):
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

    # fig = plt.figure(figsize=(10,10))

    # for i in range(len(arguments.file)):
    #    y_true[:, i][np.isnan(y_true[:, i])] = 0
    #    correlation = pearsonr(y_pred[y_true[:, i] > 0, i], y_true[y_true[:, i] > 0, i])[0]
    #    print('Correlation between true and pred n°{}:'.format(i), correlation)
    #    ax = fig.add_subplot(len(arguments.file), 1, i + 1)
    #    ax.plot(y_pred[:, i], 'b', label='prediction')
    #    ax.plot(y_true[:, i], 'r', label='experimental')
    #    ax.legend()
    #    ax.set_title('Experimental and predicted occupancy' + \
    #             'on chr 16 for model {}'.format(arguments.weight_file[6:]))
    # plt.show()


if __name__ == "__main__":
    main()
