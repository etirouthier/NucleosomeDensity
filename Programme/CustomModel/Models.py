#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:37:39 2019

@author: routhier
"""

from CustomModel.CNN_model import cnn_model as cnn
from CustomModel.CNN_LSTM_model import cnn_lstm_model as cnn_lstm
from CustomModel.CNN_dilated_model import cnn_dilated_model as cnn_dilated
from CustomModel.CNN_deep_model import cnn_deep_model as cnn_deep
from CustomModel.CNN_seq2seq_model import cnn_seq2seq_model as cnn_seq2seq
from CustomModel.CNN_try import cnn_try


def model_dictionary(num_classes=1):
    """
        Dictionary used to import the several model with parse arguments.
    """
    return {'cnn': cnn(num_classes),
            'cnn_lstm': cnn_lstm(num_classes),
            'cnn_dilated': cnn_dilated(num_classes),
            'cnn_deep': cnn_deep(num_classes),
            'cnn_seq2seq' : cnn_seq2seq(num_classes), 'cnn_try': cnn_try()}
