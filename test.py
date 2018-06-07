import tensorflow as tf
import numpy as np
import rocket_tools as rt
from keras import backend
from keras.backend import learning_phase
import cv2
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"]="1"

image_size = 128
n_images = 3
#steps = 500

tf.reset_default_graph()

#x_input = tf.placeholder(tf.float32,
#                         [None, image_size, image_size, n_images],
#                         name='input')#
'''

model = rt.model.U_NET(input_=(image_size, image_size, n_images),batch_norm=False)
y_pred = model(x_input)
'''

sess = tf.InteractiveSession()
#saver = tf.train.Saver()
saver = tf.train.import_meta_graph('./../saved_models/u_net_image_size=128_snr_all_tf_more_eph.meta')
MODEL_PATH = './../saved_models/u_net_image_size=128_snr_all_tf_more_eph'
saver.restore(sess, MODEL_PATH)
#graph = tf.get_default_graph()
#saver.restore(sess,tf.train.latest_checkpoint('./model_allbase_binary__10000'))
graph = tf.get_default_graph()
x_input=graph.get_tensor_by_name('input:0')
y_pred=graph.get_tensor_by_name('final/Sigmoid:0')
#input=graph.get_tensor_by_name('input:0')
#output=graph.get_tensor_by_name('dense/Sigmoid:0')t


snr_test = np.linspace(5, 10.0, 100)
print('start')
steps = 100
for snr in snr_test:
    results = {}
    print(snr)
    predictions = []
    true_masks = []

    image_generator = rt.image_gen.ImageGenerator(
        image_size=(image_size, image_size), snr_mean=snr, snr_std=3.0)
    batch_generator = rt.batch_generator.BatchGenerator(image_generator, batch_size=1, n_images=n_images)

    for step in range(steps):
        x_array, y_true_array = next(batch_generator)
        x_draw = x_array[0,:,:,:]*255+128
        y_in_draw =  y_true_array[0,:,:,:]*255
        cv2.imshow('x',x_draw.astype(np.uint8))
        cv2.imshow('y_in', y_in_draw.astype(np.uint8))
        #cv2.waitKey()
        y_pred_array = sess.run(y_pred, feed_dict={
            x_input: x_array, learning_phase(): 0
        })
        y_draw = y_pred_array[0, :, :, :]*255
        diff = 10*np.absolute(x_draw[:,:,0]-x_draw[:,:,2])
        cv2.imshow('diff', (diff).astype(np.uint8))
        cv2.imshow('y', (y_draw).astype(np.uint8))
        cv2.waitKey()
        predictions.append(y_pred_array)
        true_masks.append(y_true_array)

    snr_r = round(snr, 1)
    results['snr_{}'.format(snr_r)] = {
        'true': true_masks,
        'pred': predictions
    }
    np.savez_compressed('exp_res/resulst_snr_{}.npz'.format(snr_r), results=results)
print('finish')