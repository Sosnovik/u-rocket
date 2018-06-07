import tensorflow as tf
import numpy as np
import rocket_tools as rt
from keras import backend
from keras.backend import learning_phase
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def train_model(writer_path, sess,train_op,loss, model_path, steps=45000, batch_size=32, image_size=128, n_images=3):
    print 'Start'
    #x_input = tf.placeholder(tf.float32,
    #                         [None, image_size, image_size, n_images],
    #                         name='X')
    #y_true = tf.placeholder(tf.float32,
    #                        [None, image_size, image_size, n_images],
    #                        name='mask_true')
    #y_pred = model(x_input)
#
#    loss_op = -rt.losses.dice_coef(y_true, y_pred)
#    adam = tf.train.AdamOptimizer()
#    train_op = adam.minimize(loss_op)

    #with tf.variable_scope('metrics'):
    #    # tf.summary.scalar('dice', rt.losses.dice_coef(y_true, y_pred))
    #    tf.summary.scalar('cross_entropy', rt.losses.binary_crossentropy(y_true, y_pred))
    #    tf.summary.scalar('jacard', rt.losses.jacard_coef(y_true, y_pred))
    #    tf.summary.scalar('accuracy', rt.metrics.accuracy(y_true, y_pred))
    #    tf.summary.scalar('IoU', rt.metrics.IoU(y_true, y_pred))
    #    tf.summary.scalar('log_loss', loss_op)
    #    tf.summary.image('input', x_input[:1, :, :, :1])
    #    tf.summary.image('pred', y_pred[:1, :, :, :1])
    #    tf.summary.image('mask', y_true[:1, :, :, :1])
    #    summary_op = tf.summary.merge_all()
    #    writer = tf.summary.FileWriter(writer_path)

    image_generator = rt.image_gen.ImageGenerator(image_size=(image_size, image_size), snr_mean=6, snr_std=3)
    batch_generator = rt.batch_generator.BatchGenerator(image_generator, batch_size,
                                                        n_images=n_images)

    # initialize training session
    #sess = tf.InteractiveSession()
    #sess.run(tf.global_variables_initializer())

    print('started training')
    for step in range(steps):
        x_batch, y_label = next(batch_generator)
        #_, summary =
        sess.run(train_op, feed_dict={ x: x_batch, label: y_label})
        #writer.add_summary(summary, step)
        if step % 10 == 0:
            x_batch, y_label = next(batch_generator)
            print(str(step) + ' ' + str(sess.run(loss, feed_dict={x: x_batch,
                                                               label: y_label})))
        if step % 5000 == 0:
            saver = tf.train.Saver()
            Model_path = './../saved_models_n/snr_all_with_rt_steps={}'.format(step)
            saver.save(sess, Model_path)

    print('finished training')
    saver = tf.train.Saver()
    saver.save(sess, model_path)


if __name__ == "__main__":
    # parsing
    # from argparse import ArgumentParser
    # parser = ArgumentParser()

    # parser.add_argument('--snr', type=float, required=True)
    # options = parser.parse_args()

    tf.reset_default_graph()

    image_size = 128
    n_images = 3
    out_channel = 3

    TB_PATH = './../tensorboard/u_net/image_size={}_snr_all_tf_inst_norm'.format(image_size)
    MODEL_PATH = './../saved_models/u_net_image_size={}_snr_all_tf_inst_norm'.format(image_size)
    os.system('rm -rf {}'.format(TB_PATH))
    x = tf.placeholder("float", [None, image_size, image_size, 3],
                       name="input")
    output = rt.model_tf.Unet(x,out_channel, training=True)
    label = tf.placeholder("float", [None, image_size, image_size, out_channel])
    loss = tf.reduce_mean(tf.square(label - output))
    #train_opt_1 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    loss_2 = - rt.losses.dice_coef(label, output)
    adam = tf.train.AdamOptimizer()
    train_opt_2= adam.minimize(loss_2)
    #train_opt = tf.group(train_opt_1, train_opt_2)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    train_model(writer_path=TB_PATH, model_path=MODEL_PATH, sess=sess, train_op=train_opt_2, loss=loss)


    #train_model(model, writer_path=TB_PATH, model_path=MODEL_PATH, sess=sess)


#model = rt.model.U_NET(input_=(128, 128, 3))
#x_input = tf.placeholder(tf.float32, [None, 128, 128, 3],
#                             name='input')
#y_pred = model(x_input)
#sess = tf.InteractiveSession()
#saver = tf.train.Saver()
#saver.restore(sess, './../saved_models/u_net_image_size=128_snr_all_inst_norm_more_black')

#image_generator = rt.image_gen.ImageGenerator(image_size=(128, 128), snr_mean=2, snr_std=0)
#batch_generator = rt.batch_generator.BatchGenerator(image_generator, batch_size=5, n_images=3)

#x_batch, y_batch =next(batch_generator)
#pred_s = sess.run(y_pred, feed_dict={x_input: x_batch, learning_phase(): 1})