import os
import cv2

import numpy as np
import tensorflow as tf

from utility.utils import *

def create_datasets(train_json_path, train_transforms, 
                    test_json_path, test_transforms, 
                    image_size, batch_size, classes):

    train_option = {
        'json_path' : train_json_path,
        'transforms' : train_transforms,
        'image_size' : image_size,
        'batch_size' : batch_size,
        'classes' : classes,
        'shuffle' : True
    }

    test_option = {
        'json_path' : test_json_path,
        'transforms' : test_transforms,
        'image_size' : image_size,
        'batch_size' : batch_size,
        'classes' : classes,
        'shuffle' : False
    }

    return Dataset_for_classification(**train_option), Dataset_for_classification(**test_option)

class Dataset_for_classification:
    def __init__(self, json_path, image_size, batch_size, classes, transforms, shuffle):
        
        self.classes = classes
        self.batch_size = batch_size

        self.image_width, self.image_height = image_size

        self.shuffle = shuffle
        self.transforms = transforms

        self.data_dic = read_json(json_path)
        self.image_paths = list(self.data_dic.keys())

        self.batch_count = 0
        
        self.the_number_of_sample = len(self.image_paths)
        self.the_number_of_batches = self.the_number_of_sample // self.batch_size

        self.init()

    def init(self):
        self.batch_count = 0

        if self.shuffle:
            np.random.shuffle(self.image_paths)

    def preprocess_for_image(self, image_path):
        image = cv2.imread(image_path)

        if self.transforms is not None:
            image = self.transforms(image)

        return image

    def preprocess_for_label(self, label):
        label = multiple_one_hot([label], self.classes)
        return label
        
    def __iter__(self):
        return self

    def __next__(self):
        
        with tf.device('/cpu:0'):
            images = np.zeros((self.batch_size, self.image_height, self.image_width, 3), dtype=np.float32)
            labels = np.zeros((self.batch_size), dtype=np.float32)

            if self.batch_count < self.the_number_of_batches:
                count = 0
                while count < self.batch_size:
                    index = self.batch_count * self.batch_size + count
                    
                    image_path = self.image_paths[index]
                    label = self.data_dic[image_path]
                    
                    image = self.preprocess_for_image(image_path)

                    images[count] = image
                    labels[count] = label
                    count += 1

                self.batch_count += 1
                return images, labels
            else:
                self.init()
                raise StopIteration


