import tensorflow as tf
import numpy as np
from keras import backend as K
from .utils import flatten


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    #y_true_f = flatten(y_true)
    
    y_pred_f = K.flatten(y_pred)
    #y_pred_f = flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    #intersection = tf.reduce_sum(y_true_f * y_pred_f)
    
    
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)
    #return (2.0 * intersection + 1.0) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.0)
    
def dice_coef_loss(y_true, y_pred):
    return K.log(1.0-dice_coef(y_true, y_pred))



def jacard_coef(y_true, y_pred):

    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    
    return (intersection + 1.0) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection + 1.0)



def binary_crossentropy(y_true, y_pred, eps=1e-9):
    # -[true * log(pred) + (1-true) * log(1-pred)]
    y_true = flatten(y_true)
    y_pred = flatten(y_pred)
    y_pred = tf.clip_by_value(y_pred, eps, 1-eps) # save log
    
    loss = -(y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1 - y_pred))
    loss = tf.reduce_sum(loss, -1)
    loss = tf.reduce_mean(loss)
    return loss

