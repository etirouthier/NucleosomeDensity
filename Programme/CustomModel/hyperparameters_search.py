#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:44:46 2019

@author: routhier
"""

from keras.models import Sequential
from keras.layers import Dropout,Flatten, BatchNormalization
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping

import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from Programme.DataPipeline.generator import generator
from MyModuleLibrary.mykeras.losses import correlate, mae_cor, mse_var


def data():
    """
    Data providing function:
    """
    generator_train, number_of_set_train, \
    generator_val, number_of_set_val = generator('../seq_chr_sacCer3/sacCer3',
                                                 '../Start_data/proba_in_vivo.csv')
    
    return generator_train, number_of_set_train, \
    generator_val, number_of_set_val


def create_model(generator_train, number_of_set_train,
                 generator_val, number_of_set_val):
    """
    Model providing function:
    """
    window = 2001
    num_classes = 1

    fashion_model = Sequential()

    fashion_model.add(Conv2D({{choice([8, 16, 32, 64, 128])}},
                             kernel_size=({{choice([2, 4, 8, 16, 32])}},4),
                             activation='relu',
                             input_shape=(window,4,1),
                             padding='valid'))
    fashion_model.add(MaxPooling2D((choice([2, 3, 4]),1),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout({{uniform(0, 1)}}))

    fashion_model.add(Conv2D({{choice([8, 16, 32, 64, 128])}},
                             kernel_size=({{choice([2, 4, 8, 16, 32])}},1),
                             activation='relu',
                             padding='same'))
    fashion_model.add(MaxPooling2D((choice([2, 3, 4]),1),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout({{uniform(0, 1)}}))
    
    if {{choice(['add', 'not_add'])}} == 'add':
    
        fashion_model.add(Conv2D({{choice([8, 16, 32, 64, 128])}},
                                 kernel_size=({{choice([2, 4, 8, 16, 32])}},1),
                                 activation='relu',
                                 padding='same'))
        fashion_model.add(MaxPooling2D((choice([2, 3, 4]),1),padding='same'))
        fashion_model.add(BatchNormalization())
        fashion_model.add(Dropout({{uniform(0, 1)}}))

    fashion_model.add(Flatten())
    
    if {{choice(['add', 'not_add'])}} == 'add':
        fashion_model.add(Dense({{choice([10, 20, 50])}}, activation='relu'))

    fashion_model.add(Dense(num_classes, activation='relu'))

    fashion_model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                          loss={{choice([mae_cor, 'mae', 'mse', mse_var])}},
                          metrics=[correlate])

    early = EarlyStopping(monitor={{choice(['val_loss', 'val_correlate'])}},
                          min_delta=0,
                          patience=10,
                          verbose=0,
                          mode='auto')
    
    result = fashion_model.fit_generator(generator = generator_train,
                                         steps_per_epoch = 500, 
                                         epochs = 200,
                                         validation_data = generator_val, 
                                         validation_steps = 200, 
                                         callbacks = [early])
    #get the highest validation correlation of the training epochs
    val_correlate = np.amax(result.history['val_correlate']) 
    print('Best validation acc of epoch:', val_correlate)
    return {'loss': -val_correlate, 'status': STATUS_OK, 'model': fashion_model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=Trials())
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
