import tensorflow as tf

def _accuracy(y_true, y_pred):
    
    positive_pred = tf.equal(1.0, tf.round(y_pred))
    negative_pred = tf.equal(0.0, tf.round(y_pred))

    tp = tf.reduce_mean(tf.cast(positive_pred, tf.float32) * y_true)
    tn = tf.reduce_mean(tf.cast(negative_pred, tf.float32) * (1 - y_true))
    fp = tf.reduce_mean(tf.cast(positive_pred, tf.float32) * (1 - y_true))
    fn = tf.reduce_mean(tf.cast(negative_pred, tf.float32) * y_true)
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    return acc

def _IoU(y_true, y_pred):
    
    positive_pred = tf.equal(1.0, tf.round(y_pred))
    negative_pred = tf.equal(0.0, tf.round(y_pred))

    tp = tf.reduce_mean(tf.cast(positive_pred, tf.float32) * y_true)
    tn = tf.reduce_mean(tf.cast(negative_pred, tf.float32) * (1 - y_true))
    fp = tf.reduce_mean(tf.cast(positive_pred, tf.float32) * (1 - y_true))
    fn = tf.reduce_mean(tf.cast(negative_pred, tf.float32) * y_true)
    iou = 0.5 * (tp/(tp + fn + fp) + tn/(tn + fp + fn))
    
    return iou
    




    
   
    
    