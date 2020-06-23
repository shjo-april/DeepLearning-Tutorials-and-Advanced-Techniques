
import cv2
import pickle

import numpy as np

def customized_split_using_section(dataset, the_number_of_section, shuffle=True):
    if shuffle:
        np.random.shuffle(dataset)

    the_size_of_section = len(dataset) // the_number_of_section
    
    split_dataset = []
    for i in range(the_number_of_section - 1):
        split_dataset.append(dataset[:the_size_of_section])
        dataset = dataset[the_size_of_section:]
    
    split_dataset.append(dataset)

    return split_dataset

def customized_split_using_size(dataset, the_size_of_section, shuffle=True):
    if shuffle:
        np.random.shuffle(dataset)

    split_dataset = []
    while len(dataset) > 0:
        split_dataset.append(dataset[:the_size_of_section])
        dataset = dataset[the_size_of_section:]

    return split_dataset

def load_pickle(pickle_path):
    return pickle.load(open(pickle_path, 'rb'))

def dump_pickle(pickle_path, dataset):
    return pickle.dump(dataset, open(pickle_path, 'wb'))

def encode_image(image_data):
    _, image_data = cv2.imencode('.jpg', image_data)
    return image_data

def decode_image(image_data):
    image_data = np.fromstring(image_data, dtype = np.uint8)
    image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return image_data

