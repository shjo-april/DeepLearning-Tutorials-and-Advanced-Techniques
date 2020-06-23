import os
import cv2
import json

from data.writer import *
from data.utils import *
from utility.utils import *

dataset_dir = './dataset/sanghyun/'
if not os.path.isdir(dataset_dir):
    os.makedirs(dataset_dir)

tags = ['Train', 'Test']
json_paths = ['./dataset/train.json', './dataset/test.json']

for tag, json_path in zip(tags, json_paths):
    data_dic = read_json(json_path)
    image_paths = list(data_dic.keys())
    
    writer = Sanghyun_Writer(dataset_dir, tag + '_{}', 100)
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        label = data_dic[image_path]
        
        encoded_image = encode_image(image)

        key = os.path.basename(image_path)
        example = {
            'image_path' : image_path,
            'encoded_image' : encoded_image,
            'label' : label
        }
        writer(key, example)
    
    writer.save()
