import os
import cv2
import glob
import random

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
        # transforms = RandAugmentation()
        pass
    else:
        transforms = Random_HorizontalFlip()

    train_transforms = DataAugmentation(
        [
            Random_Resize(min_image_size, max_image_size),
            transforms,
            Random_Crop_with_Black((image_size, image_size))
        ]
    )

    test_transforms = DataAugmentation(
        [
            Fixed_Resize((image_size, image_size)),
            Top_Left_Crop((image_size, image_size)),
        ]
    )
    
    def decoder_func_for_training(example):
        encoded_frames = example['encoded_frames']
        label = example['label']

        indices = range(len(encoded_frames))
        indices = sorted(random.sample(indices, args.the_number_of_frame))

        frames = [decode_image(encoded_frames[index]) for index in indices]
        frames = train_transforms(frames)

        return frames, label

    def decoder_func_for_testing(example):
        encoded_frames = example['encoded_frames']
        label = example['label']
        
        length = len(encoded_frames)
        indices = list(range(0, length, length // args.the_number_of_frame))[:args.the_number_of_frame]

        frames = [decode_image(encoded_frames[index]) for index in indices]
        frames = train_transforms(frames)

        return frames, label

    the_number_of_cpu_cores = mp.cpu_count()

    train_reader_option = {
        'sanghyun_paths' : glob.glob('./dataset/{}/sanghyun/Train*'.format(args.dataset_name)),
        'training' : True,
        'drop_remainder' : True,
        'batch_size' : args.batch_size,
        'image_shape' : (args.image_size, args.image_size, 3),
        
        'the_number_of_loader' : 1, 
        'the_number_of_decoder' : the_number_of_cpu_cores // 2, 
        
        'decode_fn' : decoder_func_for_testing,
        'names' : ['image', 'label']
    }
    train_reader = Sanghyun_Reader(**train_reader_option)

    train_reader.start()

    class_names = [name.strip() for name in open('./dataset/{}/class_names.txt'.format(args.dataset_name)).readlines()]
    class_names = np.asarray(class_names)

    for batch_frames, batch_labels in train_reader:
        
        for frames, label in zip(batch_frames, batch_labels):
            
            class_name = class_names[int(label)]

            for index, frame in enumerate(frames):
                cv2.imwrite('./frame={}.jpg'.format(index + 1), frame)
            input(class_name)

        input()

    train_reader.close()
