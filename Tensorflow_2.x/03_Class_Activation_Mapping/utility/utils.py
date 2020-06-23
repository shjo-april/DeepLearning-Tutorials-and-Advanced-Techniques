# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import cv2
import time
import json
import pickle
import datetime

import numpy as np
import xml.etree.ElementTree as ET

def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    probs = probs.transpose((2, 0, 1))

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    Q = np.array(Q).reshape((n_labels, h, w))
    return Q.transpose((1, 2, 0))

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

def read_xml(xml_path, class_names):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)
    
    bboxes = []
    classes = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        if not label in class_names:
            continue
            
        bbox = obj.find('bndbox')
        
        bbox_xmin = max(min(int(bbox.find('xmin').text.split('.')[0]), image_width - 1), 0)
        bbox_ymin = max(min(int(bbox.find('ymin').text.split('.')[0]), image_height - 1), 0)
        bbox_xmax = max(min(int(bbox.find('xmax').text.split('.')[0]), image_width - 1), 0)
        bbox_ymax = max(min(int(bbox.find('ymax').text.split('.')[0]), image_height - 1), 0)

        if (bbox_xmax - bbox_xmin) == 0 or (bbox_ymax - bbox_ymin) == 0:
            continue
        
        bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        classes.append(label)

    return np.asarray(bboxes, dtype = np.float32), np.asarray(classes)

def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d-%02dh%02dm%02ds" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s

def log_print(string, log_path = './log.txt'):
    print(string)
    
    if log_path is not None:
        f = open(log_path, 'a+')
        f.write(string + '\n')
        f.close()

def csv_print(data_list, log_path = './log.csv'):
    string = ''
    for data in data_list:
        if type(data) != type(str):
            data = str(data)
        string += (data + ',')
    
    if log_path is not None:
        f = open(log_path, 'a+')
        f.write(string + '\n')
        f.close()

def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def write_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent = '\t')

def multiple_one_hot(labels, classes):
    v = np.zeros([classes], dtype = np.float32)
    for label in labels:
        v[label] = 1.
    return v

