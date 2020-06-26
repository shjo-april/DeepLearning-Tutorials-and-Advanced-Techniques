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

    for epoch in range(50):
        train_reader.start()

        for images, labels in train_reader:
            # print(images.shape, labels.shape)
            pass

        train_reader.close()

        test_reader.start()

        for images, labels in test_reader:
            pass

        test_reader.close()

        print('epoch={}'.format(epoch + 1))
        # input()