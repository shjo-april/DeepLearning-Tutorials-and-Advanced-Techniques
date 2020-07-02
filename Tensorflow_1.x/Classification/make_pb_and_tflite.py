
import os
import cv2
import glob

import numpy as np
import tensorflow as tf

import multiprocessing as mp

from core.model import *
from core.config import *
from core.augmentors import *
from core.utils import *

from data.reader import *
from data.utils import *

from utility.utils import *
from utility.timer import *

if __name__ == '__main__':
    args = get_config()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu

    gpu_id = int(args.use_gpu)
    experiments_dir = './experiments/{}/'.format(args.dataset_name)

    class_names = [line.strip() for line in open('./dataset/{}/class_names.txt'.format(args.dataset_name)).readlines()]
    classes = len(class_names)
    
    pb_dir = create_directory(experiments_dir + 'pb/')
    ckpt_dir = create_directory(experiments_dir + 'ckpt/')
    tflite_dir = create_directory(experiments_dir + 'tflite/')

    tensorboard_dir = create_directory(experiments_dir + 'tensorboard/')
    
    pb_name = 'model.pb'
    tflite_name = 'model.tflite'

    test_image_var = tf.placeholder(tf.float32, [None, args.image_size, args.image_size, 3], name='images')
    
    test_option = {
        'is_training' : False,
        'classes' : classes
    }

    with tf.device(tf.DeviceSpec(device_type = "GPU", device_index = 0)):
        with tf.variable_scope(tf.get_variable_scope(), reuse = False):
            test_logits_op = EfficientNet_Lite(**test_option)(test_image_var)

    # Create session and saver
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep = 2)

    # Create ckpt file to pb file.
    line = open(ckpt_dir + 'checkpoint', 'r').readlines()[1]
    model_name = line.strip().replace('all_model_checkpoint_paths: ', '').replace('\"', '')

    best_valid_path = ckpt_dir + model_name
    
    saver.restore(sess, best_valid_path)
    create_pb_file(sess, pb_dir, pb_name)

    print('[i] Create pb file. ({})'.format(pb_dir + pb_name))

    # Create pb file to tflite file.
    pb_path = pb_dir + pb_name

    input_arrays = ['images']
    output_arrays = ['Classifier/predictions']

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(pb_path, input_arrays, output_arrays)
    tflite_model = converter.convert()

    open(tflite_dir + tflite_name, "wb").write(tflite_model)

    print('[i] Create tflite file. ({})'.format(tflite_dir + tflite_name))


