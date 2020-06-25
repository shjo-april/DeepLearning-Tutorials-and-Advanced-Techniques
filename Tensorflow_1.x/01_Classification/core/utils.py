
import tensorflow as tf

def normalize(cams):
    min_values = tf.math.reduce_min(cams, axis=(1, 2), keepdims=True)
    max_values = tf.math.reduce_max(cams, axis=(1, 2), keepdims=True)

    cams = (cams - min_values) / (max_values - min_values + 1e-5)
    return cams

def get_model_vars(scope = None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)