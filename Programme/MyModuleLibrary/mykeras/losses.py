# -*- coding: utf8 -*-

"""
	This module contains the custom losses or metrics that can be used to train or to evaluate a neural network.
	It is made to work as a usual loss or metric.
"""

try:
    from keras import backend as K
except ModuleNotFoundError:
    from tensorflow.keras import backend as K


def correlate(y_true, y_pred):
    """
		Calculate the correlation between the predictions and the labels.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = correlate)
		>>> load_model('file', custom_objects = {'correlate : correlate})
	"""
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)
    
    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))
    
    return sigma_XY/(sigma_X*sigma_Y + K.epsilon())

def mae_cor(y_true, y_pred):
    """
	   Calculate the mean absolute error minus the correlation between
        predictions and  labels.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = mae_cor)
		>>> load_model('file', custom_objects = {'mae_cor : mae_cor})
	"""
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)
    
    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))
    
    cor = sigma_XY/(sigma_X*sigma_Y + K.epsilon())
    mae = K.mean(K.abs(y_true - y_pred))
    
    return 1 + mae - cor

def mse_var(y_true, y_pred) : 
    """
		Calculate the mean squared error between the predictions and the labels and add the absolute difference of
		variance between the distribution of labels and the distribution of predictions.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = mse_var)
		>>> load_model('file', custom_objects = {'mse_var' : mse_var})
	"""
    X = y_true - y_pred
    
    Y = K.mean(X**2) + K.abs(K.var(y_true) - K.var(y_pred))
    
    return Y

def bray_curtis(y_true, y_pred) :
    """
		Calculate the Bray Curtis distance between the predictions and the label.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = bray_curtis)
		>>> load_model('file', custom_objects = {'bray_curtis : bray_curtis})
	"""
    X = K.sum(K.minimum(y_true, y_pred))
    
    Y = K.sum(y_true + y_pred)
    
    return (1 - 2*X/Y)

def mae_wo_zeros(y_true, y_pred) :
    """
		Calculate the mean absolute error between the predictions and the label but without taking into account the contribution
		of zeros within a sequence.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = mae_wo_zeros)
		>>> load_model('file', custom_objects = {'mae_wo_zeros : mae_wo_zeros})

		..notes:: It is equivalent to set sequence weight to the sign of the sequence. A method already exist in keras but does not
				  seem to work.
	"""    
    X = y_true - y_pred
    
    sample_weight = K.sign(y_true)
    
    X_weighted = sample_weight*K.abs(X)
    
    Y = K.mean(X_weighted)
    
    return Y

def mse_wo_zeros_var(y_true, y_pred) :
    """
		Calculate the mean absolute error between the predictions and the label but without taking into account the contribution
		of zeros within a sequence. After that it adds the absolute value of the difference of variance between the prediction
		distribution and the label distribution.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = mse_wo_zeros_var)
		>>> load_model('file', custom_objects = {'mse_wo_zeros_var : mse_wo_zeros_var})

		..notes:: It is equivalent to set sequence weight to the sign of the sequence and to use mse_var. A method already exists
				  in keras but does not seem to work.
	"""        
    X = y_true - y_pred
    
    sample_weight = K.sign(y_true)
    
    X_weighted = sample_weight*K.abs(X)
    
    Y = K.mean(X_weighted**2) + K.abs(K.var(y_true) - K.var(y_pred))
    
    return Y

def MCC(y_true, y_pred):
     """
    		Calculate the Mattheew correlation coefficient between the predictions and the label.
    
    		:Example:
    
    		>>> model.compile(optimizer = 'adam', losses = MCC)
    		>>> load_model('file', custom_objects = {'MCC : MCC})
    
    		..notes:: This metrics is usefull to evaluate the accuracy with imbalanced dataset.
     """        
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
    
     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
     
     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg)
    
     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg)
    
     numerator = (tp * tn - fp * fn)
     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
     return numerator / (denominator + K.epsilon())
    
    

    































