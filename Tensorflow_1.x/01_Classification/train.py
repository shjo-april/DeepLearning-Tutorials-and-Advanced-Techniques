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

    pb_dir = create_directory(experiments_dir + 'pb/')
    ckpt_dir = create_directory(experiments_dir + 'ckpt/')
    tflite_dir = create_directory(experiments_dir + 'tflite/')

    tensorboard_dir = create_directory(experiments_dir + 'tensorboard/')

    pb_name = '{}.pb'.format(args.dataset_name)
    tflite_name = '{}.tflite'.format(args.dataset_name)

    # Define training dataset and testing dataset.
    image_size = args.image_size
    max_image_size = int(image_size * 1.25)
    min_image_size = max_image_size // 2

    if args.augmentation == 'RandAugment':
        transforms = RandAugmentation()
    else:
        transforms = Random_HorizontalFlip()

    train_transforms = DataAugmentation(
        [
            Random_Resize(min_image_size, max_image_size),
            transforms,
            Random_Crop_with_Black((image_size, image_size))
        ]
    )
    
    def decoder_func(example):
        image = example['encoded_image']
        label = example['label']

        image = decode_image(image)
        image = train_transforms(image)

        return image, label

    the_number_of_cpu_cores = mp.cpu_count()

    train_reader_option = {
        'sanghyun_paths' : glob.glob('./dataset/{}/sanghyun/Train*'.format(args.dataset_name)),
        'training' : True,
        'drop_remainder' : True,
        'batch_size' : args.batch_size,
        'image_shape' : (args.image_size, args.image_size, 3),
        
        'the_number_of_loader' : 1, 
        'the_number_of_decoder' : the_number_of_cpu_cores // 2, 
        
        'decode_fn' : decoder_func,
        'names' : ['image', 'label']
    }
    train_reader = Sanghyun_Reader(**train_reader_option)

    test_reader_option = {
        'sanghyun_paths' : glob.glob('./dataset/{}/sanghyun/Test*'.format(args.dataset_name)),
        'training' : False,
        'drop_remainder' : False,
        'batch_size' : args.batch_size,
        'image_shape' : (args.image_size, args.image_size, 3),
        
        'the_number_of_loader' : 1, 
        'the_number_of_decoder' : the_number_of_cpu_cores // 2, 
        
        'decode_fn' : decoder_func,
        'names' : ['image', 'label']
    }
    test_reader = Sanghyun_Reader(**test_reader_option)

    # Build the model for training.
    train_option = {
        'is_training' : True,
        'classes' : args.classes
    }

    train_image_var = tf.placeholder(tf.float32, [None, args.image_size, args.image_size, 3])
    train_label_var = tf.placeholder(tf.int64, [None])

    with tf.device(tf.DeviceSpec(device_type = "GPU", device_index = gpu_id)):
        with tf.variable_scope(tf.get_variable_scope(), reuse = False):
            train_model = EfficientNet_Lite(**train_option)
            train_logits_op = train_model(train_image_var)

    class_loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = train_logits_op, labels = train_label_var))

    train_vars = get_model_vars()
    l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in train_vars if 'kernel' in var.name]) * args.weight_decay

    loss_op = class_loss_op + l2_reg_loss_op

    correct_op = tf.equal(tf.argmax(train_logits_op, axis = 1), train_label_var)
    train_accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

    # Define learning rate schedule for training
    global_step = tf.placeholder(dtype = tf.int32)

    if args.learning_rate_schedule == 'cosine_annealing':
        warmup_epochs = int(0.1 * args.max_epochs)

        warmup_lr_op = tf.to_float(global_step) / tf.to_float(warmup_epochs) * args.learning_rate
        decay_lr_op = 0.5 * args.learning_rate * (1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / args.max_epochs))

        learning_rate = tf.where(global_step < warmup_epochs, warmup_lr_op, decay_lr_op)
    else:
        learning_rate = args.learning_rate

    # Define Optimizer for training the model.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        if args.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9, use_nesterov = True)
        elif args.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)

        train_op = optimizer.minimize(loss_op, colocate_gradients_with_ops=True)

    # Build the model for testing.
    test_image_var = tf.placeholder(tf.float32, [None, args.image_size, args.image_size, 3], name='images')
    test_label_var = tf.placeholder(tf.int64, [None])

    test_option = {
        'is_training' : False,
        'classes' : args.classes
    }

    with tf.device(tf.DeviceSpec(device_type = "GPU", device_index = gpu_id)):
        with tf.variable_scope(tf.get_variable_scope(), reuse = True):
            test_logits_op = EfficientNet_Lite(**test_option)(test_image_var)

    correct_op = tf.equal(tf.argmax(test_logits_op, axis = 1), test_label_var)
    test_accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100
    
    # Build history of the model for monitoring
    train_summary_dic = {
        'losses/total_loss' : loss_op,
        'losses/class_loss' : class_loss_op,
        'losses/l2_regularization' : l2_reg_loss_op,

        'monitors/Train_Accuracy' : train_accuracy_op,
        'monitors/learning_rate' : learning_rate,
    }
    train_summary_op = tf.summary.merge([tf.summary.scalar(name, train_summary_dic[name]) for name in train_summary_dic.keys()])
    
    train_writer = tf.summary.FileWriter(tensorboard_dir)
    print('[i] tensorboard directory is {}'.format(tensorboard_dir))

    # Create session and saver
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(max_to_keep = 2)

    # Restore pretrained weights from ImageNet
    pretrained_vars = [var for var in train_vars if train_model.pretrained_model_name in var.name]

    imagenet_saver = tf.train.Saver(var_list = pretrained_vars)
    imagenet_saver.restore(sess, '../Pretrained_models/{}/model.ckpt'.format(train_model.pretrained_model_name))

    # training
    timer = Timer()

    iteration = 0

    best_valid_accuracy = -1
    best_valid_path = None

    # for epoch in range(args.max_epochs):
        
    #     train_loss = []
    #     train_accuracy = []
        
    #     timer.tik()
    #     train_reader.start()

    #     for images, labels in train_reader:
    #         _, loss, accuracy, summary = sess.run([train_op, loss_op, train_accuracy_op, train_summary_op], feed_dict={train_image_var : images, train_label_var : labels, global_step:epoch})

    #         train_loss.append(loss)
    #         train_accuracy.append(accuracy)

    #         train_writer.add_summary(summary, iteration); iteration += 1
        
    #     train_sec = timer.tok()
    #     train_reader.close()
        
    #     train_loss = np.mean(train_loss)
    #     train_accuracy = np.mean(train_accuracy)
        
    #     print('[i] epoch={}, iteration={}, loss={:.6f}, accuracy={:.2f}%, {}sec'.format(epoch + 1, iteration, train_loss, train_accuracy, train_sec))

    #     if (epoch + 1) % 5 == 0:
    #         valid_accuracy = []
            
    #         timer.tik()
    #         test_reader.start()

    #         for images, labels in test_reader:
    #             accuracy = sess.run(test_accuracy_op, feed_dict={test_image_var : images, test_label_var : labels})
    #             valid_accuracy.append(accuracy)

    #         test_sec = timer.tok()
    #         test_reader.close()
            
    #         valid_accuracy = np.mean(valid_accuracy)
    #         print('[i] epoch={}, valid_accuracy={:.2f}%, best_valid_accuracy={:.2f}%, {}sec'.format(epoch + 1, valid_accuracy, best_valid_accuracy, test_sec))
    
    #         if valid_accuracy > best_valid_accuracy:
    #             best_valid_accuracy = valid_accuracy
    #             best_valid_path = ckpt_dir + '{}.ckpt'.format(epoch + 1) 

    #             saver.save(sess, best_valid_path)

    # saver.save(sess, ckpt_dir + 'end.ckpt')

    # Create ckpt file to pb file.
    saver.restore(sess, best_valid_path)
    create_pb_file(sess, pb_dir, pb_name)

    print('[i] Create pb file. ({})'.format(pb_dir + pb_name))
    
    # Create ckpt file to tflite file.
    pb_path = pb_dir + pb_name
    
    input_arrays = ['images']
    output_arrays = ['Classifier/predictions']

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(pb_path, input_arrays, output_arrays)
    tflite_model = converter.convert()

    open(tflite_dir + tflite_name, "wb").write(tflite_model)

    print('[i] Create tflite file. ({})'.format(tflite_dir + tflite_name))


