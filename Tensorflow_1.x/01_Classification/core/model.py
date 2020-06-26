
import numpy as np
import tensorflow as tf

import core.efficientnet.efficientnet_builder as efficientnet
import core.efficientnet.efficientnet_lite_builder as efficientnet_lite

class EfficientNet_Lite:
    def __init__(self, is_training, classes):
        self.is_training = is_training
        self.classes = classes
        
        '''
        'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
        'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
        'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
        'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
        'efficientnet-lite4': (1.4, 1.8, 300, 0.3),
        '''
        self.model_name = 'efficientnet-lite0'
        self.pretrained_model_name = self.model_name
        
        self.mean = tf.constant(efficientnet.MEAN_RGB)
        self.std  = tf.constant(efficientnet.STDDEV_RGB)

        self.width_coeff, self.depth_coeff, self.image_size, self.drop_rate = efficientnet_lite.efficientnet_lite_params(self.model_name)
    
    def __call__(self, x):
        x = (x[..., ::-1] - self.mean) / self.std
        _, x = efficientnet_lite.build_model_base(x, self.model_name, self.is_training)
        
        with tf.variable_scope('Classifier'):
            x = tf.reduce_mean(x['reduction_5'], axis=[1, 2])
            logits_op = tf.layers.dense(x, units=self.classes, name='logits')
            predictions_op = tf.nn.softmax(logits_op, name='predictions')

        return logits_op
