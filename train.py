import tensorflow as tf
import numpy as np
from keras.backend import learning_phase
import rocket_tools as rt
import os


def train_model(model, snr, writer_path, steps=5000, batch_size=64, image_size=128, n_images=3):    
    x_input = tf.placeholder(tf.float32, 
                             [None, image_size, image_size, n_images], 
                             name='input')
    y_true = tf.placeholder(tf.float32, 
                            [None, image_size, image_size, n_images], 
                            name='mask_true')
    y_pred = model(x_input) 
    
    loss_op = -rt.losses.dice_coef(y_true, y_pred)
    adam = tf.train.AdamOptimizer()
    train_op = adam.minimize(loss_op)
    
    with tf.variable_scope('metrics'):
        tf.summary.scalar('dice', rt.losses.dice_coef(y_true, y_pred))
        tf.summary.scalar('jacard', rt.losses.jacard_coef(y_true, y_pred)) 
        tf.summary.scalar('accuracy', rt.metrics.accuracy(y_true, y_pred))
        tf.summary.scalar('IoU', rt.metrics.IoU(y_true, y_pred))
        tf.summary.scalar('log_loss', loss_op)
        tf.summary.image('input', x_input[:1, :, :, :1])
        tf.summary.image('pred', y_pred[:1, :, :, :1])
        tf.summary.image('mask', y_true[:1, :, :, :1])
        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(writer_path)
    
    
    
    image_generator = rt.image_gen.ImageGenerator(image_size=(image_size, image_size))
    batch_generator = rt.batch_generator.BatchGenerator(image_generator, batch_size, 
                                                        snr=snr, n_images=n_images)
    
    # initialize training session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
   
    print('started training')
    for step in range(steps):
        x, y = next(batch_generator)
        _, summary = sess.run([train_op, summary_op], 
                              feed_dict={
                                  x_input: x, 
                                  y_true: y, 
                                  learning_phase(): 1
                              })
        writer.add_summary(summary, step) 
        
    print('finished training')
        

        
        
if __name__ == "__main__":
    # parsing
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--snr', type=float, required=True)
    options = parser.parse_args()
      
    snr = options.snr
    image_size = 128
    n_images = 3

    TB_PATH = './tensorboard/u_net/image_size={}_snr={}'.format(image_size, snr)
    MODEL_PATH = './saved_models/u_net_image_size={}_snr={}'.format(image_size, snr)
    os.system('rm -rf {}'.format(TB_PATH))
    
    model = rt.model.U_NET(input_=(image_size, image_size, n_images))

    train_model(model, writer_path=TB_PATH, snr=snr)

    saver = tf.train.Saver()
    saver.save(sess, MODEL_PATH)

        
        
        
        
