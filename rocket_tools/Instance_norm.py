import tensorflow as tf


def instance_norm(x, training):
   if  training == True:
        var_shape = (x.get_shape().as_list()[-1],)
        mu, sigma_2 = tf.nn.moments(x, [1, 2], keep_dims=True)

        shift = tf.get_variable('shift', dtype=tf.float32,
                                initializer=tf.zeros(var_shape))

        scale = tf.get_variable('scale', dtype=tf.float32,
                                initializer=tf.ones(var_shape))

        normalized = (x - mu) / (sigma_2 + 1e-3)**(0.5)
        x = scale * normalized + shift
    
    return x


    
