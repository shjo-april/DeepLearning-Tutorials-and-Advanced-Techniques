import os
import cv2
import copy

import numpy as np
import multiprocessing as mp

# try:
#     os.environ['Sanghyun']
# except KeyError:
#     import tensorflow as tf

from utility.utils import *

def create_datasets_from_images(train_json_path, train_transforms, 
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

    return Dataset_for_classification_from_images(**train_option), Dataset_for_classification_from_images(**test_option)

def create_datasets_from_pickles(train_pickle_paths, train_transforms, 
                    test_pickle_paths, test_transforms, 
                    image_size, batch_size, selected_pickle_size, use_cores, max_size):
    
    train_option = {
        'pickle_paths' : train_pickle_paths,
        'transforms' : train_transforms,
        'image_size' : image_size,
        'batch_size' : batch_size,
        'shuffle' : True,
        'selected_pickle_size' : selected_pickle_size,
        'drop_remainder' : True,
        'use_cores' : use_cores,
        'max_size' : max_size,
    }

    test_option = {
        'pickle_paths' : test_pickle_paths,
        'transforms' : test_transforms,
        'image_size' : image_size,
        'batch_size' : batch_size,
        'shuffle' : False,
        'selected_pickle_size' : selected_pickle_size,
        'drop_remainder' : False,
        'use_cores' : use_cores,
        'max_size' : max_size,
    }

    return Dataset_for_classification_from_pickle_files(**train_option), Dataset_for_classification_from_pickle_files(**test_option)

class Dataset_for_classification_from_images:
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

class Prefetch_using_queue:
    def __init__(self, class_func, pickle_paths, option, use_cores, max_size):
        self.main_queue = mp.Queue(maxsize = max_size)

        self.instances = []
        self.split_pickle_paths = custom_split(pickle_paths, use_cores)
        for i in range(use_cores):
            option['pickle_paths'] = self.split_pickle_paths[i]
            self.instances.append(class_func(self.main_queue, option))

    def start(self):
        for instance in self.instances:
            instance.start()

            print(instance)

    def join(self):
        for instance in self.instances:
            instance.join()

    def get(self):
        return self.main_queue.get()

    def get_size(self):
        return self.main_queue.qsize()

class Decoder(mp.Process):
    def __init__(self, queue, option):
        super().__init__()
        self.daemon = True
        self.queue = queue

        self.batch_size = option['batch_size']
        self.selected_pickle_size = option['selected_pickle_size']

        self.transforms = option['transforms']
        self.drop_remainder = option['drop_remainder']

        self.image_width, self.image_height = option['image_size']

        self.pickle_paths = option['pickle_paths']
        if option['shuffle']: np.random.shuffle(self.pickle_paths)

    def preprocess_for_image(self, encoded_image):
        image = decode_image(encoded_image)

        if self.transforms is not None:
            image = self.transforms(image)

        return image

    def generate_batch_data(self, batch_dataset):
        images = np.zeros((len(batch_dataset), self.image_height, self.image_width, 3), dtype=np.float32)
        labels = np.zeros((len(batch_dataset)), dtype=np.float32)

        for count, (encoded_image, label) in enumerate(batch_dataset[:self.batch_size]):
            image = self.preprocess_for_image(encoded_image)

            images[count] = image
            labels[count] = label
        
        return images, labels
    
    def run(self):
        dataset = []
        while len(self.pickle_paths) > 0:
            sub_pickle_paths = self.pickle_paths[:self.selected_pickle_size]
            self.pickle_paths = self.pickle_paths[self.selected_pickle_size:]
            
            # 1. load dataset
            for pickle_path in sub_pickle_paths:
                for image_name, encoded_image, label in load_pickle(pickle_path):
                    dataset.append([encoded_image, label])

            np.random.shuffle(dataset)

            # 2. decode dataset and generate batch dataset
            while len(dataset) >= self.batch_size:
                self.queue.put(list(self.generate_batch_data(dataset[:self.batch_size])))
                dataset = dataset[self.batch_size:]
        
        if len(dataset) > 0 and not self.drop_remainder:
            self.queue.put(list(self.generate_batch_data(dataset)))

        self.queue.put(StopIteration)

class Dataset_for_classification_from_pickle_files(mp.Process):
    def __init__(self, pickle_paths, image_size, batch_size, transforms, shuffle, selected_pickle_size, drop_remainder, use_cores, max_size):
        self.use_cores = use_cores
        self.max_size = max_size

        self.pickle_paths = pickle_paths
        self.option = {
            'selected_pickle_size' : selected_pickle_size, 
            'drop_remainder' : drop_remainder, 
            
            'image_size' : image_size, 
            'batch_size' : batch_size, 
            'transforms' : transforms, 
            
            'shuffle' : shuffle
        }
        
    def init(self):
        self.decoder = Prefetch_using_queue(Decoder, self.pickle_paths, self.option, self.use_cores, self.max_size)
        self.decoder.start()

    def join(self):
        self.decoder.join()

    def __iter__(self):
        return self

    def __next__(self):
        data = self.decoder.get()

        if data == StopIteration:
            raise StopIteration
        else:
            return data
