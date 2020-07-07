
import numpy as np
import tensorflow as tf

import core.efficientnet.efficientnet_builder as efficientnet
import core.efficientnet.efficientnet_lite_builder as efficientnet_lite

# 1. 3D-VGG16-BN
# 2. EfficientNet-ConvLSTM
# 3. I3D

class VGG16_BN_3D:
    def __init__(self, is_training, classes):
        self.is_training = is_training
        self.classes = classes

        self.mean = tf.constant(efficientnet.MEAN_RGB)
        self.std  = tf.constant(efficientnet.STDDEV_RGB)

    def conv_bn_relu(self, x, filters, kernel_size):
        x = tf.layers.conv3d(x, filters, kernel_size, padding='same')
        x = tf.layers.batch_normalization(x, training=self.is_training)
        x = tf.nn.relu(x)
        return x

    def __call__(self, x, initial_feature_size=32):
        x = (x[..., ::-1] - self.mean) / self.std

        with tf.variable_scope('block1'):
            x = self.conv_bn_relu(x, initial_feature_size, [3, 3, 3])
            x = self.conv_bn_relu(x, initial_feature_size, [3, 3, 3])
            x = tf.layers.max_pooling3d(x, [1, 2, 2], [1, 2, 2])
            initial_feature_size *= 2
            print(x)

        with tf.variable_scope('block2'):
            x = self.conv_bn_relu(x, initial_feature_size, [3, 3, 3])
            x = self.conv_bn_relu(x, initial_feature_size, [3, 3, 3])
            x = tf.layers.max_pooling3d(x, [1, 2, 2], [1, 2, 2])
            initial_feature_size *= 2
            print(x)

        with tf.variable_scope('block3'):
            x = self.conv_bn_relu(x, initial_feature_size, [3, 3, 3])
            x = self.conv_bn_relu(x, initial_feature_size, [3, 3, 3])
            x = tf.layers.max_pooling3d(x, [1, 2, 2], [1, 2, 2])
            initial_feature_size *= 2
            print(x)

        with tf.variable_scope('block4'):
            x = self.conv_bn_relu(x, initial_feature_size, [3, 3, 3])
            x = self.conv_bn_relu(x, initial_feature_size, [3, 3, 3])
            x = tf.layers.max_pooling3d(x, [1, 2, 2], [1, 2, 2])
            print(x)

        with tf.variable_scope('block5'):
            x = self.conv_bn_relu(x, initial_feature_size, [3, 3, 3])
            x = self.conv_bn_relu(x, initial_feature_size, [3, 3, 3])
            x = tf.layers.max_pooling3d(x, [1, 2, 2], [1, 2, 2])
            print(x)

        with tf.variable_scope('Classifier'):
            x = tf.reduce_mean(x, axis=[1, 2, 3])
            logits_op = tf.layers.dense(x, units=self.classes, name='logits')
            predictions_op = tf.nn.softmax(logits_op, name='predictions')

        return logits_op

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
