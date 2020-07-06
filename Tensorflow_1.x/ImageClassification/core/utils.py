
import tensorflow as tf

def normalize(cams):
    min_values = tf.math.reduce_min(cams, axis=(1, 2), keepdims=True)
    max_values = tf.math.reduce_max(cams, axis=(1, 2), keepdims=True)

    cams = (cams - min_values) / (max_values - min_values + 1e-5)
    return cams

def get_model_vars(scope = None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def create_pb_file(sess, pb_dir, pb_name):
    from tensorflow.python.platform import app
    from tensorflow.python.summary import summary
    from tensorflow.python.framework import graph_util

    gd = sess.graph.as_graph_def()
    converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, ['Classifier/predictions'])
    tf.train.write_graph(converted_graph_def, pb_dir, pb_name, as_text=False)

    