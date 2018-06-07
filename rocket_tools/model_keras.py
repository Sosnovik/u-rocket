from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Reshape, Dropout, Conv2DTranspose, BatchNormalization, Concatenate, Add, Multiply
from keras.models import Model
from keras import backend as K
import numpy as np
import cv2




def conv_bn(m, dim, acti, bn):
    m = Conv2D(dim, 3, activation=acti, padding='same')(m)
    return BatchNormalization()(m) if bn else m

def level_block(m, dim, depth, inc_rate, acti, dropout, bn, fcn):
    if depth > 0:
        n = conv_bn(m, dim, acti, bn)
        n = Dropout(dropout)(n) if dropout else n
        n = conv_bn(n, dim, acti, bn)
        m = Conv2D(dim, 3, strides=2, activation=acti, padding='same')(n) if fcn else MaxPooling2D()(n)
        m = level_block(m, int(inc_rate*dim), depth-1, inc_rate, acti, dropout, bn, fcn)
        m = UpSampling2D()(m)
        m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        m = Concatenate(axis=3)([n, m])
    m = conv_bn(m, dim, acti, bn)
    return conv_bn(m, dim, acti, bn)



def UNet(img_shape, start_ch=64, depth=4, inc_rate=2, activation='relu', dropout=0.05, bn=True, fcn=False):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, bn, fcn)
    output = Conv2D(3, 1, activation='sigmoid')(o)
    return Model(inputs=i, outputs=output)
