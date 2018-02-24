{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "from keras.models import Model\n",
    "from keras.layers import *\n",
    "from keras import backend as K\n",
    "from keras.optimizers import Adam\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def dice_coef(y_true, y_pred):\n",
    "    y_true_f = K.flatten(y_true)\n",
    "    y_pred_f = K.flatten(y_pred)\n",
    "    intersection = K.sum(y_true_f * y_pred_f)\n",
    "    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)\n",
    "\n",
    "\n",
    "def jacard_coef(y_true, y_pred):\n",
    "    y_true_f = K.flatten(y_true)\n",
    "    y_pred_f = K.flatten(y_pred)\n",
    "    intersection = K.sum(y_true_f * y_pred_f)\n",
    "    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)\n",
    "\n",
    "\n",
    "def jacard_coef_loss(y_true, y_pred):\n",
    "    return -jacard_coef(y_true, y_pred)\n",
    "\n",
    "\n",
    "def dice_coef_loss(y_true, y_pred):\n",
    "    return -dice_coef(y_true, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def gaussian2D(shape, mean, cov):\n",
    "    '''\n",
    "    add gaussian to source image\n",
    "    params:\n",
    "        shape - shape of the output\n",
    "        mean - array-like tuple of x_mean, y_mean - relative positions\n",
    "        cov - 2x2 covariance matrix\n",
    "        magnitude - maximal intensity of gaussian\n",
    "    '''\n",
    "    X, Y = np.indices(shape)\n",
    "    X = (X + 1).astype(np.float) / (shape[0] + 1)\n",
    "    Y = (Y + 1).astype(np.float) / (shape[1] + 1)\n",
    "    r = np.stack([X, Y], -1) - mean\n",
    "    exp = np.einsum('ijk,kl,ijl->ij', r, np.linalg.inv(cov), r)\n",
    "    exp = np.exp(-0.5 * exp)\n",
    "    return exp\n",
    "\n",
    "def random_small_rotation():\n",
    "    angle = np.random.normal(0, scale=0.1)\n",
    "    matrix = np.array([\n",
    "        [np.cos(angle), np.sin(angle)],\n",
    "        [-1.0 * np.sin(angle), np.cos(angle)]\n",
    "    ])\n",
    "    return matrix\n",
    "\n",
    "def add_gaussian_on_image(image, magnitude, mean, radius, cov):        \n",
    "    new_image = image.copy()\n",
    "    gauss = gaussian2D(image.shape, mean, radius * cov)\n",
    "    new_image += gauss * magnitude\n",
    "    return new_image\n",
    "\n",
    "def gen_random_images():\n",
    "\n",
    "    # Background\n",
    "    dark_color = np.random.randint(50, 150)\n",
    "    light_color = np.random.randint(1, 255 - dark_color)\n",
    "    \n",
    "    img = get_random_background() * dark_color\n",
    "    t_vec = np.random.random(size=2)\n",
    "    t_vec = t_vec / np.linalg.norm(t_vec)\n",
    "    \n",
    "    images = []\n",
    "    masks = []\n",
    "    \n",
    "    \n",
    "    point = np.random.random(size=2)\n",
    "    radius = np.random.uniform(0.0001, 0.0025)\n",
    "    cov = np.eye(2) + 0.5 * np.random.uniform(-1, 1, size=(2, 2))\n",
    "    \n",
    "    for _ in range(N_IMAGES):\n",
    "        generated_img = add_gaussian_on_image(img, light_color, point, radius, cov)\n",
    "        images.append(generated_img)\n",
    "        \n",
    "        mask = add_gaussian_on_image(np.zeros_like(img), 1.0, point, radius, cov) > 0.5\n",
    "        mask = mask.astype('float32')\n",
    "        masks.append(mask)\n",
    "        \n",
    "        t_vec = random_small_rotation().dot(t_vec)\n",
    "        distance = 0.1 * np.random.normal(1, scale=0.1)\n",
    "        point += t_vec * distance\n",
    "        cov = random_small_rotation().dot(cov)\n",
    "        radius *= np.random.uniform(0.7, 0.9)\n",
    "        \n",
    "    # White noise\n",
    "    noise = np.random.randint(0, 200, size=image_size)\n",
    "    p_noise = np.random.random(size=image_size)\n",
    "    th = np.random.uniform(0, 0.1)\n",
    "    for img in images:\n",
    "#         pass\n",
    "        img[p_noise < th] = noise[p_noise < th]\n",
    "#         new_images.append(np.expand_dims(img[i], -1\n",
    "    \n",
    "    return np.stack(images, axis=-1), np.stack(masks, axis=-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def double_conv_layer(x, size, dropout, batch_norm):\n",
    "    axis = 3\n",
    "    conv = Conv2D(size, (3, 3), padding='same')(x)\n",
    "    if batch_norm is True:\n",
    "        conv = BatchNormalization(axis=axis)(conv)\n",
    "    conv = Activation('relu')(conv)\n",
    "    conv = Conv2D(size, (3, 3), padding='same')(conv)\n",
    "    if batch_norm is True:\n",
    "        conv = BatchNormalization(axis=axis)(conv)\n",
    "    conv = Activation('relu')(conv)\n",
    "    if dropout > 0:\n",
    "        conv = Dropout(dropout)(conv)\n",
    "    return conv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def ZF_UNET_224(dropout_val=0.0, batch_norm=True):\n",
    "    inputs = Input((512, 512, 3))\n",
    "    axis = 3\n",
    "    filters = 16\n",
    "\n",
    "    conv_224 = double_conv_layer(inputs, filters, dropout_val, batch_norm)\n",
    "    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)\n",
    "\n",
    "    conv_112 = double_conv_layer(pool_112, 2*filters, dropout_val, batch_norm)\n",
    "    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)\n",
    "\n",
    "    conv_56 = double_conv_layer(pool_56, 4*filters, dropout_val, batch_norm)\n",
    "    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)\n",
    "\n",
    "    conv_28 = double_conv_layer(pool_28, 8*filters, dropout_val, batch_norm)\n",
    "    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)\n",
    "\n",
    "    conv_14 = double_conv_layer(pool_14, 16*filters, dropout_val, batch_norm)\n",
    "    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)\n",
    "\n",
    "    conv_7 = double_conv_layer(pool_7, 32*filters, dropout_val, batch_norm)\n",
    "\n",
    "    up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=axis)\n",
    "    up_conv_14 = double_conv_layer(up_14, 16*filters, dropout_val, batch_norm)\n",
    "\n",
    "    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=axis)\n",
    "    up_conv_28 = double_conv_layer(up_28, 8*filters, dropout_val, batch_norm)\n",
    "\n",
    "    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=axis)\n",
    "    up_conv_56 = double_conv_layer(up_56, 4*filters, dropout_val, batch_norm)\n",
    "\n",
    "    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=axis)\n",
    "    up_conv_112 = double_conv_layer(up_112, 2*filters, dropout_val, batch_norm)\n",
    "\n",
    "    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=axis)\n",
    "    up_conv_224 = double_conv_layer(up_224, filters, 0, batch_norm)\n",
    "\n",
    "    conv_final = Conv2D(3, (1, 1))(up_conv_224)\n",
    "    conv_final = BatchNormalization(axis=axis)(conv_final)\n",
    "    conv_final = Activation('sigmoid')(conv_final)\n",
    "\n",
    "    model = Model(inputs, conv_final, name=\"ZF_UNET_224\")\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
